import time
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import test_single_volume  # 使用已修正的指标计算
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_synapse import InSARSegDataset
# 你原来导入的一些包（transforms/vutils/plt）在本文件并未使用，可按需保留

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
parser.add_argument('--volume_path', type=str, default='E:/PycharmProjects/TransUNet/data/Hephaestus/images')
parser.add_argument('--list_dir', type=str, default='E:/PycharmProjects/TransUNet/lists/lists_hephaestus')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--is_savenii', action="store_true")
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
parser.add_argument('--test_save_dir', type=str, default='../predictions')
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vit_patches_size', type=int, default=16)
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = InSARSegDataset(
        image_root=args.volume_path,
        mask_root="E:/PycharmProjects/TransUNet/data/Hephaestus/masks",
        split="test",
        transform=None,
        list_txt_root=args.list_dir
    )

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    # 累加按类的指标和（形状：(C-1, 6)），最终再除以样本数
    sum_metrics = None
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        # 数据集输出是 [B,H,W,C]，这里转为 [B,C,H,W]
        image = sampled_batch["image"].permute(0, 3, 1, 2)
        label = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]

        metric_i = test_single_volume(
            image, label, model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=1
        )

        metric_i = np.array(metric_i, dtype=np.float64)  # (C-1, 6)
        if sum_metrics is None:
            sum_metrics = metric_i
        else:
            sum_metrics += metric_i

        dice_vals, hd95_vals, iou_vals, prec_vals, recall_vals, f1_vals = zip(*metric_i)
        logging.info(
            'idx %d case %s | Dice %.4f | HD95 %.4f | IoU %.4f | P %.4f | R %.4f | F1 %.4f' %
            (i_batch, case_name,
             np.mean(dice_vals), np.mean(hd95_vals), np.mean(iou_vals),
             np.mean(prec_vals), np.mean(recall_vals), np.mean(f1_vals))
        )

    # 样本平均
    metric_mean = sum_metrics / max(1, len(db_test))  # (C-1, 6)

    # 按类打印（跳过背景类 0）
    for i in range(1, args.num_classes):
        mean_dice, mean_hd95, mean_iou, mean_p, mean_r, mean_f1 = metric_mean[i - 1]
        logging.info('Mean class %d | Dice %.4f | HD95 %.4f | IoU %.4f | P %.4f | R %.4f | F1 %.4f' %
                     (i, mean_dice, mean_hd95, mean_iou, mean_p, mean_r, mean_f1))

    # 所有前景类再做一次平均
    mean_vals = np.mean(metric_mean, axis=0)
    mean_dice, mean_hd95, mean_iou, mean_p, mean_r, mean_f1 = mean_vals
    logging.info(
        'Final Performance (avg over foreground classes): '
        'Dice %.4f | HD95 %.4f | IoU %.4f | P %.4f | R %.4f | F1 %.4f' %
        (mean_dice, mean_hd95, mean_iou, mean_p, mean_r, mean_f1)
    )


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (args.img_size // args.vit_patches_size,
                                   args.img_size // args.vit_patches_size)
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # 你训练好的模型权重
    snapshot = "E:/PycharmProjects/TransUNet/model/TU_InSAR7_R50-ViT-B_16_224_0901_1716(DUTransUNet)/epoch_149.pth"
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = "epoch_149"

    # 日志目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_folder = f'./test_log/test_log_{snapshot_name}_{timestamp}'
    os.makedirs(log_folder, exist_ok=True)
    log_file_name = f"{snapshot_name}_{timestamp}.txt"
    logging.basicConfig(filename=os.path.join(log_folder, log_file_name), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # 结果保存目录
    args.test_save_dir = './predictions'
    timestamp2 = time.strftime("%m%d_%H%M")
    snapshot_name = snapshot_name + f"_{timestamp2}"
    test_save_path = os.path.join(args.test_save_dir, snapshot_name)
    os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, test_save_path)
