import argparse
import json
import os
from datetime import datetime
from torch import optim
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
import random
from src.data.collation import Collator
from src.data.dataset import Twitter_Dataset
from src.data.tokenization_new import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.MAESC_model import MultiModalBartModel_AESC
from src.model.model import TRCPretrain
from src.training import fine_tune
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
from src.model.metrics import AESCSpanMetric
from src.model.generater import SequenceGeneratorModel
import src.eval_utils as eval_utils
import numpy as np
import torch.backends.cudnn as cudnn
import src.resnet.resnet as resnet
from src.resnet.resnet_utils import myResnet
import collections

def main(rank, args):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, timestamp)
    args.checkpoint_path=checkpoint_path
    tb_writer = None
    add_name = ''
    log_dir = os.path.join(args.log_dir, timestamp + add_name)

    # make log dir and tensorboard writer if log_dir is specified
    if args.log_dir is not None:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'),
                    enabled=True)

    # make checkpoint dir if not exist
    if args.is_check == 1 and not os.path.isdir(checkpoint_path) and not args.no_train:
        os.makedirs(checkpoint_path)
        logger.info('Made checkpoint directory: "{}"'.format(checkpoint_path))

    logger.info('Initialed with {} GPU(s)'.format(args.gpu_num), pad=True)
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))

    # =========================== model =============================

    logger.info('Loading model...')

    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        device = torch.device("cuda:{}".format(rank))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    args.device=device
    tokenizer = ConditionTokenizer(args=args)
    label_ids = list(tokenizer.mapping2id.values())
    senti_ids = list(tokenizer.senti2id.values())

    if args.model_config is not None:
        bart_config = MultiModalBartConfig.from_dict(
            json.load(open(args.model_config)))
    else:
        bart_config = MultiModalBartConfig.from_pretrained(args.checkpoint)

    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout

    bos_token_id = 0  # 因为是特殊符号
    eos_token_id = 1

    # resnet
    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load('/home/zhouru/ABSA4/src/resnet/resnet152.pth'))
    img_encoder = myResnet(net, True, device)
    img_encoder.to(device)

    if args.checkpoint and args.no_train==False:
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,args.bart_model, tokenizer,label_ids)
        model = SequenceGeneratorModel(seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
        if args.trc_on:
            trc_pretrain_model=TRCPretrain.from_pretrained(
                args.trc_pretrain_file,
                config=bart_config,
                bart_model=args.bart_model,
                tokenizer=tokenizer,
                label_ids=label_ids,
                senti_ids=senti_ids,
                args=args,
                error_on_mismatch=False)
            if args.encoder=='trc':
                model.seq2seq_model.encoder.load_state_dict(trc_pretrain_model.encoder.state_dict())
            model.seq2seq_model.noun_linear.load_state_dict(trc_pretrain_model.noun_linear.state_dict())
            model.seq2seq_model.multi_linear.load_state_dict(trc_pretrain_model.multi_linear.state_dict())
            model.seq2seq_model.att_linear.load_state_dict(trc_pretrain_model.att_linear.state_dict())
            model.seq2seq_model.linear.load_state_dict(trc_pretrain_model.linear.state_dict())
            model.seq2seq_model.alpha_linear1.load_state_dict(trc_pretrain_model.alpha_linear1.state_dict())
            model.seq2seq_model.alpha_linear2.load_state_dict(trc_pretrain_model.alpha_linear2.state_dict())
            logger.info('trc model loaded.')
    else:
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        model = SequenceGeneratorModel(seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    scaler = GradScaler() if args.amp else None

    logger.info('Loading data...')
    collate_aesc = Collator(tokenizer,
                            aesc_enabled=True,
                            text_only=args.text_only)

    train_dataset = Twitter_Dataset(args.img_path,args.dataset[0][1], split='train')

    dev_dataset = Twitter_Dataset(args.img_path,args.dataset[0][1], split='dev')
    test_dataset = Twitter_Dataset(args.img_path,args.dataset[0][1], split='test')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              collate_fn=collate_aesc)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=collate_aesc)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=collate_aesc)

    callback = None
    metric = AESCSpanMetric(eos_token_id,
                            num_labels=len(label_ids),
                            conflict_id=-1,
                            dataset=args.dataset[0][0])


    if args.no_train:
        if args.dataset[0][0] == 'twitter15':
            model=torch.load('AoM-ckpt/Twitter2015/AoM.pt').to(device)
        else:
            model=torch.load('AoM-ckpt/Twitter2017/AoM.pt').to(device)

        res_test = eval_utils.eval(args, model, img_encoder, test_loader, metric, device)
        logger.info('TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
            res_test['aesc_pre'], res_test['aesc_rec'], res_test['aesc_f']))
    else:
        model.train()
        img_encoder.train()
        img_encoder.zero_grad()
        fine_tune(epochs=args.epochs,
                  model=model,
                  img_encoder=img_encoder,
                  train_loader=train_loader,
                  dev_loader=dev_loader,
                  test_loader=test_loader,
                  metric=metric,
                  optimizer=optimizer,
                  args=args,
                  device=device,
                  logger=logger,
                  callback=callback,
                  log_interval=1,
                  tb_writer=tb_writer,
                  tb_interval=1,
                  scaler=scaler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        action='append',
                        nargs=2,
                        metavar=('DATASET_NAME', 'DATASET_PATH'),
                        required=True,
                        help='')
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        help=' ')
    # required

    parser.add_argument('--checkpoint_dir',
                        required=True,
                        type=str,
                        help='where to save the checkpoint')
    parser.add_argument('--bart_model',
                        default='facebook/bart-base',
                        type=str,
                        help='bart pretrain model')
    # path
    parser.add_argument(
        '--log_dir',
        default=None,
        type=str,
        help='path to output log files, not output to file if not specified')
    parser.add_argument('--model_config',
                        default=None,
                        type=str,
                        help='path to load model config')
    parser.add_argument('--text_only',
                        default=False,
                        type=bool,
                        help='if only input the text')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='name or path to load weights')
    parser.add_argument('--lr_decay_every',
                        default=4,
                        type=int,
                        help='lr_decay_every')
    parser.add_argument('--lr_decay_ratio',
                        default=0.8,
                        type=float,
                        help='lr_decay_ratio')
    # training and evaluation
    parser.add_argument('--epochs',
                        default=35,
                        type=int,
                        help='number of training epoch')
    parser.add_argument('--eval_every', default=1, type=int, help='eval_every')
    parser.add_argument('--eval_step', default=50, type=int, help='eval_step')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--num_beams',
                        default=4,
                        type=int,
                        help='level of beam search on validation')
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup')
    parser.add_argument(
        '--dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the transformer. This overwrites the model config')
    parser.add_argument(
        '--classif_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the classification layers. This overwrites the model config'
    )
    parser.add_argument(
        '--attention_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the attention layers. This overwrites the model config'
    )
    parser.add_argument(
        '--activation_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the activation layers. This overwrites the model config'
    )

    # hardware and performance
    parser.add_argument('--grad_clip', default=5, type=float, help='grad_clip')
    parser.add_argument('--gpu_num',
                        default=1,
                        type=int,
                        help='number of GPUs in total')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port',
                        type=str,
                        default='12355',
                        help='master port for DDP')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='training batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='#workers for data loader')
    parser.add_argument('--max_len', type=int, default=10, help='max_len')
    parser.add_argument('--max_len_a',
                        type=float,
                        default=0.6,
                        help='max_len_a')

    parser.add_argument('--bart_init',
                        type=int,
                        default=1,
                        help='use bart_init or not')

    parser.add_argument('--check_info',
                        type=str,
                        default='',
                        help='check path to save')
    parser.add_argument('--is_check',
                        type=int,
                        default=1,
                        help='save the model or not')
    parser.add_argument('--task', type=str, default='', help='task type')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help=' ')
    parser.add_argument('--no_train',
                        action='store_true',
                        help=' ')
    parser.add_argument('--trc_pretrain_file',
                        default='/home/zhouru/ABSA3/checkpoint_dir/2022-09-20-16-12-27/model60/pytorch_model.bin',
                        type=str,
                        help=' ')
    parser.add_argument('--trained_file',
                        default='/home/zhouru/ABSA4/train17/2022-11-23-16-49-35/pytorch_model.bin',
                        type=str,
                        help=' ')
    parser.add_argument('--senti_pretrain_file',
                        default='/home/zhouru/ABSA4/checkpoint_dir/2022-11-30-11-04-51/model45_minloss/pytorch_model.bin',
                        type=str,
                        help=' ')
    parser.add_argument('--encoder',
                        default=None,
                        type=str,
                        help=' ')
    parser.add_argument('--sentinet_on',
                        action='store_true',
                        help=' ')
    parser.add_argument('--nn_attention_on',
                        action='store_true'
                        )

    parser.add_argument('--nn_attention_mode',
                        type=int,
                        default=0,
                        )
    parser.add_argument('--trc_on',
                        action='store_true'
                        )
    parser.add_argument('--gcn_on',
                        action='store_true',
                        help=' ')
    parser.add_argument('--gcn_dropout',
                        type=float,
                        default=0
                        )
    parser.add_argument('--gcn_proportion',
                        type=float,
                        default=0.5)
    parser.add_argument('--dep_mode',
                        type=int,
                        default=0,
                        )
    args = parser.parse_args()
    if args.encoder=='trc':
        args.trc_on=True

    dep_list=['text_cosine','text_cat_sim','text_cos_img_noun_sim']
    args.dep_mode=dep_list[args.dep_mode]

    nn_attention_list=['cat','multi-head','cos_']
    args.nn_attention_mode=nn_attention_list[args.nn_attention_mode]

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError(
            '--model_config and --checkpoint cannot be empty at the same time')
    args.img_path=''
    if args.dataset[0][0]=='twitter15':
        args.img_path='/home/zhouru/IJCAI2019_data/twitter2015_images'
    elif args.dataset[0][0]=='twitter17':
        args.img_path='/home/zhouru/IJCAI2019_data/twitter2017_images'
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    main(args.rank, args)
