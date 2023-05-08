import argparse
import json
import os
from datetime import datetime
import numpy as np
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
from src.data.dataset import TRC_Dataset
from src.data.tokenization_new import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.model import TRCPretrain,MultiModalBartModelForPretrain
from src.training import trc_pretrain
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
import torch.backends.cudnn as cudnn
DATASET_NAMES = ('TRC', )
import src.resnet.resnet as resnet
from src.resnet.resnet_utils import myResnet

def main(rank, args):

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(args.checkpoint_dir, timestamp)
    tb_writer = None
    log_dir = os.path.join(args.log_dir, timestamp)

    # make log dir and tensorboard writer if log_dir is specified
    if  args.log_dir is not None:
        os.makedirs(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'),
                    enabled=True)

    # make checkpoint dir if not exist
    if  not os.path.isdir(checkpoint_path):
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
    tokenizer = ConditionTokenizer(args)
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

    if args.checkpoint:
        pretrain_model = MultiModalBartModelForPretrain.from_pretrained(
            args.checkpoint,
            config=bart_config,
            bart_model=args.bart_model,
            tokenizer=tokenizer,
            label_ids=label_ids,
            senti_ids=senti_ids,
            args=args,
            error_on_mismatch=False)
        model = TRCPretrain.from_pretrained(
            args.checkpoint,
            config=bart_config,
            bart_model=args.bart_model,
            tokenizer=tokenizer,
            label_ids=label_ids,
            senti_ids=senti_ids,
            args=args,
            error_on_mismatch=False)
        model.encoder.load_state_dict(pretrain_model.encoder.state_dict())
    else:
        model = TRCPretrain(bart_config, args.bart_model,
                                               tokenizer, label_ids, senti_ids,
                                               args)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    scaler = GradScaler() if args.amp else None
    # =========================== data =============================

    logger.info('Loading data...')
    collate = Collator(tokenizer,
                             mlm_enabled=False,
                             senti_enabled=False,
                             ae_enabled=False,
                             oe_enabled=False,
                             aesc_enabled=False,
                             anp_enabled=False,
                             trc_enabled=True,)


    TRC_data = None
    for name, path in args.dataset:
        if name == 'TRC':
            TRC_data = TRC_Dataset(path)
    start = datetime.now()

    # ========================== training ============================
    logger.info('Start training', pad=True)
    scaler = GradScaler() if args.amp else None

    dataloader=DataLoader(dataset=TRC_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=True,
                          collate_fn=collate
                          )

    # resnet
    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load('/home/zhouru/ABSA2/src/resnet/resnet152.pth'))
    img_encoder = myResnet(net, True, device)
    img_encoder.to(device)
    args.checkpoint_path=checkpoint_path
    trc_pretrain(epochs=args.epochs,
             model=model,
             img_encoder=img_encoder,
             train_loader=dataloader,
             optimizer=optimizer,
             args=args,
             device=device,
             logger=logger,
             log_interval=1,
             tb_writer=tb_writer,
             tb_interval=1,
             scaler=scaler)


    if not args.cpu:
        cleanup_process()


def parse_args():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--dataset',
                        action='append',
                        nargs=2,
                        metavar=('DATASET_NAME', 'DATASET_PATH'),
                        required=True,
                        help='append a dataset, one of "{}"'.format(
                            '", "'.join(DATASET_NAMES)))
    parser.add_argument('--checkpoint_dir',
                        required=True,
                        type=str,
                        help='where to save the checkpoint')
    parser.add_argument('--bart_model',
                        default='facebook/bart-base',
                        type=str,
                        help='bart pretrain model')

    parser.add_argument('--checkpoint_every',
                        default=20,
                        type=int,
                        help='checkpoint_every')

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
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='name or path to load weights')

    # model
    parser.add_argument('--no_event',
                        dest='use_event',
                        action='store_false',
                        help='not to use event descriptions')
    parser.add_argument('--no_image',
                        dest='use_image',
                        action='store_false',
                        help='not to use image features')

    parser.add_argument('--no_rp',
                        dest='rp_enabled',
                        action='store_false',
                        help='do not use relation prediction')
    parser.add_argument('--epochs',
                        default=60,
                        type=int,
                        help='number of training epoch')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--num_gen',
                        default=1,
                        type=int,
                        help='number of generated sentence on validation')
    parser.add_argument('--num_beams',
                        default=1,
                        type=int,
                        help='level of beam search on validation')
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument(
        '--validate_loss',
        action='store_true',
        help='compute the validation loss at the end of each epoch')
    parser.add_argument(
        '--validate_score',
        action='store_true',
        help=
        'compute the validation score (BLEU, METEOR, etc.) at the end of each epoch'
    )
    parser.add_argument('--max_img_num',
                        type=int,
                        default=49,
                        help='max number of image feature per data entry')
    parser.add_argument(
        '--lm_max_len',
        type=int,
        default=30,
        help='max number of words for the language modeling per data entry')
    parser.add_argument('--mrm_probability',
                        type=float,
                        default=0.15,
                        help='mask probability for MRM')
    parser.add_argument('--mlm_probability',
                        type=float,
                        default=0.15,
                        help='mask probability for MLM')

    # dropout
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
                        default=32,
                        help='training batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='#workers for data loader')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--ANP_loss_type',
                        type=str,
                        default='KL',
                        help='ANP_loss_type')
    parser.add_argument('--mlm_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--senti_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--trc_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--anp_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--anp_generate_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--ae_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--oe_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--ae_oe_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--mrm_enabled',
                        type=int,
                        default=1,
                        help='mlm_enabled')
    parser.add_argument('--mrm_loss_type',
                        type=str,
                        default='KL',
                        help='mrm_loss_type')
    parser.add_argument('--bart_init', type=int, default=1, help='bart_init')
    parser.add_argument('--task', type=str, default='', help='task type')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help=' ')
    # parser.set_defau  lts()
    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError(
            '--model_config and --checkpoint cannot be empty at the same time')

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    # mp.spawn(main, args=(args, ), nprocs=args.gpu_num, join=True)
    main(args.rank, args)
