from datetime import datetime
import numpy as np
from torch.cuda.amp import autocast
import src.model.utils as utils
import src.eval_utils as eval_utils
import src.eval_utils as eval_utils
import torch
import os
from src.utils import save_training_data


def fine_tune(epochs,
              model,
              img_encoder,
              train_loader,
              dev_loader,
              test_loader,
              metric,
              optimizer,
              device,
              args,
              logger=None,
              callback=None,
              log_interval=1,
              tb_writer=None,
              tb_interval=1,
              scaler=None):

    total_step = len(train_loader)*epochs
    model.train()
    img_encoder.train()
    img_encoder.zero_grad()
    total_loss = 0
    epoch=0
    global_step=0
    start_time = datetime.now()
    best_dev_res = None
    best_dev_test_res = None
    best_test_res = None
    eval_step=100
    while epoch < epochs:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        for i, batch in enumerate(train_loader):
            model.train()
            img_encoder.train()
            img_encoder.zero_grad()

            # Forward pass
            global_step+=1
            aesc_infos = {key: value for key, value in batch['AESC'].items()}
            with torch.no_grad():
                imgs_f=[x.numpy().tolist() for x in batch['image_features']]
                imgs_f=torch.tensor(imgs_f).to(device)
                imgs_f, img_mean, img_att = img_encoder(imgs_f)
                img_att=img_att.view(-1, 2048, 49).permute(0, 2, 1)
            with autocast(enabled=args.amp):
                loss = model.forward(
                    input_ids=batch['input_ids'].to(device),
                    image_features=list(map(lambda x: x.to(device), img_att)),
                    sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
                    noun_mask=batch['noun_mask'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    dependency_matrix=batch['dependency_matrix'].to(device),
                    aesc_infos=aesc_infos)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, args.epochs, epoch*len(train_loader) + i + 1, total_step, loss.item()))
            # Backward and optimize

            cur_step = i + 1 + epoch * total_step
            t_step = args.epochs * total_step
            liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
            utils.set_lr(optimizer, liner_warm_rate * args.lr)

            optimizer.zero_grad()

            loss.backward()
            utils.clip_gradient(optimizer, args.grad_clip)

            optimizer.step()

            # test
            if (global_step + 1) % eval_step == 0:
                eval_step=args.eval_step
                logger.info('Step {}'.format(global_step + 1), pad=True)

                res_dev = eval_utils.eval(args, model,img_encoder ,dev_loader, metric, device)

                res_test = eval_utils.eval(args, model,img_encoder, test_loader, metric, device)

                logger.info('DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                    res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f']))
                logger.info('DEV  ae_p:{} ae_r:{} ae_f:{}'.format(
                    res_dev['ae_pre'], res_dev['ae_rec'], res_dev['ae_f']))
                logger.info('DEV  sc_acc:{} sc_r:{} sc_f:{}'.format(
                    res_dev['sc_acc'], res_dev['sc_rec'], res_dev['sc_f']))

                logger.info('TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                    res_test['aesc_pre'], res_test['aesc_rec'], res_test['aesc_f']))
                logger.info('TEST  ae_p:{} ae_r:{} ae_f:{}'.format(
                    res_test['ae_pre'], res_test['ae_rec'], res_test['ae_f']))
                logger.info('TEST  sc_acc:{} sc_r:{} sc_f:{}'.format(
                    res_test['sc_acc'], res_test['sc_rec'], res_test['sc_f']))

                save_flag = False
                if best_dev_res is None:
                    best_dev_res = res_dev
                    best_dev_test_res = res_test
                else:
                    if best_dev_res['aesc_f'] < res_dev['aesc_f']:
                        best_dev_res = res_dev
                        best_dev_test_res = res_test

                if best_test_res is None:
                    best_test_res = res_test
                    save_flag = True
                else:
                    if best_test_res['aesc_f'] < res_test['aesc_f']:
                        best_test_res = res_test
                        save_flag = True

                if args.is_check == 1 and save_flag:
                    current_checkpoint_path = os.path.join(args.checkpoint_path,
                                                           args.check_info)
                    model.seq2seq_model.save_pretrained(current_checkpoint_path)
                    save_img_encoder(args,img_encoder)
                    torch.save(img_encoder, os.path.join(args.checkpoint_path, 'resnet152.pt'))
                    torch.save(model,os.path.join(args.checkpoint_path,'AoM.pt'))
                    logger.info('save model to {} !!!!!!!!!!!'.format(current_checkpoint_path))
        epoch += 1

    logger.info("Training complete in: " + str(datetime.now() - start_time),pad=True)
    logger.info('---------------------------')
    logger.info('BEST DEV:-----')
    logger.info('BEST DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_res['aesc_pre'], best_dev_res['aesc_rec'],
        best_dev_res['aesc_f']))
    logger.info('BEST DEV  ae_p:{} ae_r:{} ae_f:{}'.format(
        best_dev_res['ae_pre'], best_dev_res['ae_rec'],
        best_dev_res['ae_f']))
    logger.info('BEST DEV  sc_acc:{} sc_r:{} sc_f:{}'.format(
        best_dev_res['sc_acc'], best_dev_res['sc_rec'],
        best_dev_res['sc_f']))

    logger.info('BEST DEV TEST:-----')
    logger.info('BEST DEV--TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_test_res['aesc_pre'], best_dev_test_res['aesc_rec'],
        best_dev_test_res['aesc_f']))
    logger.info('BEST DEV--TEST  ae_p:{} ae_r:{} ae_f:{}'.format(
        best_dev_test_res['ae_pre'], best_dev_test_res['ae_rec'],
        best_dev_test_res['ae_f']))
    logger.info('BEST DEV--TEST  sc_acc:{} sc_r:{} sc_f:{}'.format(
        best_dev_test_res['sc_acc'], best_dev_test_res['sc_rec'],
        best_dev_test_res['sc_f']))

    logger.info('BEST TEST:-----')
    logger.info('BEST TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_test_res['aesc_pre'], best_test_res['aesc_rec'],
        best_test_res['aesc_f']))
    logger.info('BEST TEST  ae_p:{} ae_r:{} ae_f:{}'.format(
        best_test_res['ae_pre'], best_test_res['ae_rec'],
        best_test_res['ae_f']))
    logger.info('BEST TEST  sc_acc:{} sc_r:{} sc_f:{}'.format(
        best_test_res['sc_acc'], best_test_res['sc_rec'],
        best_test_res['sc_f']))


def trc_pretrain(epochs,
             model,
             img_encoder,
             train_loader,
             optimizer,
             device,
             args,
             logger=None,
             callback=None,
             log_interval=1,
             tb_writer=None,
             tb_interval=1,
             scaler=None):
    start=datetime.now()
    total_step = len(train_loader)*epochs
    model.train()
    img_encoder.train()
    img_encoder.zero_grad()
    min_loss = 100
    epoch=0
    global_step = 0
    criterion=torch.nn.CrossEntropyLoss(reduction='mean')
    start_time = datetime.now()
    while epoch<epochs:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        for i, batch in enumerate(train_loader):
            # Forward pass
            global_step+=1
            with torch.no_grad():
                imgs_f=[x.numpy().tolist() for x in batch['image_features']]
                imgs_f=torch.tensor(imgs_f).to(device)
                imgs_f, img_mean, img_att = img_encoder(imgs_f)
                img_att=img_att.view(-1, 2048, 49).permute(0, 2, 1)
            with autocast(enabled=args.amp):
                logits = model.forward(
                    input_ids=batch['input_ids'].to(device),
                    image_features=list(map(lambda x: x.to(device), img_att)),
                    noun_mask=batch['noun_mask'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
                loss=criterion(logits.view(-1,2),torch.tensor(batch['ifpairs']).to(args.device))
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()


        logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs,loss.item()))
        if loss.item() < min_loss:
            min_loss=loss.item()
            current_checkpoint_path = os.path.join(
                args.checkpoint_path, ('model{}_minloss').format(epoch))
            model.save_pretrained(current_checkpoint_path)
            save_training_data(path=current_checkpoint_path,
                               optimizer=optimizer,
                               scaler=scaler,
                               epoch=epoch)
            logger.info('Saved checkpoint at "{}"'.format(args.checkpoint_path))
        # save checkpoint
        elif epoch % args.checkpoint_every == 0:
            if args.bart_init == 0:
                current_checkpoint_path = os.path.join(
                    args.checkpoint_path, 'model{}random_again'.format(epoch))
            else:
                current_checkpoint_path = os.path.join(
                    args.checkpoint_path, ('model{}').format(epoch))
            if args.cpu:
                model.save_pretrained(current_checkpoint_path)
            else:
                model.save_pretrained(current_checkpoint_path)
            save_training_data(path=current_checkpoint_path,
                               optimizer=optimizer,
                               scaler=scaler,
                               epoch=epoch)
            logger.info('Saved checkpoint at "{}"'.format(args.checkpoint_path))
        epoch += 1
    logger.info("Finish pretraining  " + str(datetime.now() - start), pad=True)



def save_finetune_model(model):
    torch.save(model.state_dict(),'/home/zhouru/ABSA3/save_model/best_model.pth')


def save_img_encoder(args,img_encoder):
    file_name=os.path.join(args.checkpoint_path,'resnet152.pth')
    torch.save(img_encoder.state_dict(),file_name)
    pass