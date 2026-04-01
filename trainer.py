import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, FocalLoss
from torchvision import transforms
from datasets.dataset_synapse import InSARSegDataset, RandomGenerator


# 全局初始化 args（需在函数外声明）
args = None
# 全局变量存储 seed
global_seed = None


def apply_cutmix(image_batch, label_batch, alpha=1.0):
    '''CutMix增强，image_batch: [B,C,H,W], label_batch: [B,H,W]'''
    B, C, H, W = image_batch.size()
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(B).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(H, W, lam)
    new_image = image_batch.clone()
    new_label = label_batch.clone()
    new_image[:, :, bbx1:bbx2, bby1:bby2] = image_batch[rand_index, :, bbx1:bbx2, bby1:bby2]
    new_label[:, bbx1:bbx2, bby1:bby2] = label_batch[rand_index, bbx1:bbx2, bby1:bby2]

    return new_image, new_label

def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_mixup(image_batch, label_batch, alpha=0.4):
    '''MixUp增强，image_batch: [B,C,H,W], label_batch: [B,H,W]'''
    B, C, H, W = image_batch.size()
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(B).cuda()

    mixed_image = lam * image_batch + (1 - lam) * image_batch[rand_index, :]
    label_a, label_b = label_batch, label_batch[rand_index]

    return mixed_image, label_a, label_b, lam



def worker_init_fn(worker_id):
    if global_seed is not None:
        random.seed(global_seed + worker_id)
    else:
        random.seed(1234 + worker_id)  # 默认值，防止未初始化

def trainer_synapse(args_, model, snapshot_path):
    global args  # 声明全局变量
    args = args_  # 将参数赋给全局变量

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = InSARSegDataset(
        image_root=args.volume_path,
        mask_root="E:/PycharmProjects/TransUNet/data/Hephaestus/masks",
        split="train",
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]),
        list_txt_root=args.list_dir
    )

    print("The length of train set is: {}".format(len(db_train)))

    global global_seed
    global_seed = args.seed

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = FocalLoss(alpha=1.0, gamma=2.0)
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            prob = random.random()
            if prob < 0.25:
                image_batch, label_batch = apply_cutmix(image_batch, label_batch)
            elif 0.25 <= prob < 0.5:
                image_batch, label_a, label_b, lam = apply_mixup(image_batch, label_batch)

            if prob < 0.25:  # CutMix
                outputs, aux_outputs = model(image_batch)
                loss_ce_main = ce_loss(outputs, label_batch.long())
                loss_ce_aux = ce_loss(aux_outputs, label_batch.long())
                soft_outputs = torch.softmax(outputs, dim=1)
                loss_dice = dice_loss(soft_outputs, label_batch, softmax=False)
                total_loss = 0.5 * loss_ce_main + 0.3 * loss_ce_aux + 1.0 * loss_dice

                loss_ce = loss_ce_main  # 用于日志输出



            elif 0.25 <= prob < 0.5:  # MixUp
                outputs, aux_outputs = model(image_batch)
                loss_ce_main = lam * ce_loss(outputs, label_a.long()) + (1 - lam) * ce_loss(outputs, label_b.long())
                loss_ce_aux = lam * ce_loss(aux_outputs, label_a.long()) + (1 - lam) * ce_loss(aux_outputs,
                                                                                               label_b.long())
                soft_outputs = torch.softmax(outputs, dim=1)
                loss_dice = lam * dice_loss(soft_outputs, label_a, softmax=False) + (1 - lam) * dice_loss(soft_outputs,
                                                                                                          label_b,
                                                                                                          softmax=False)
                total_loss = 0.5 * loss_ce_main + 0.3 * loss_ce_aux + 1.0 * loss_dice

                loss_ce = loss_ce_main  # 用于日志输出



            else:  # Normal
                outputs, aux_outputs = model(image_batch)
                loss_ce_main = ce_loss(outputs, label_batch.long())
                loss_ce_aux = ce_loss(aux_outputs, label_batch.long())
                soft_outputs = torch.softmax(outputs, dim=1)
                loss_dice = dice_loss(soft_outputs, label_batch, softmax=False)
                total_loss = 0.5 * loss_ce_main + 0.3 * loss_ce_aux + 1.0 * loss_dice

                loss_ce = loss_ce_main  # 用于日志输出

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', total_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : total_loss : %.4f | ce: %.4f | dice: %.4f' % (
                iter_num, total_loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                pred_output = torch.softmax(outputs, dim=1)
                pred_output = torch.argmax(pred_output, dim=1, keepdim=True)
                writer.add_image('train/Prediction', pred_output[1, ...] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"