import time
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
from datasets.dataset_synapse import InSARSegDataset, RandomGenerator
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='E:/PycharmProjects/TransUNet/data/Hephaestus')
parser.add_argument('--volume_path', type=str, default='E:/PycharmProjects/TransUNet/data/Hephaestus/images')
parser.add_argument('--list_dir', type=str, default='E:/PycharmProjects/TransUNet/lists/lists_hephaestus')
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment name')
parser.add_argument('--num_classes', type=int, default=2, help='number of segmentation classes')
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
parser.add_argument('--vit_patches_size', type=int, default=16)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--n_gpu', type=int, default=1, help='number of gpus to use')
args = parser.parse_args()


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

    # prepare model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.attention_type = 'cbam'
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()


    time_suffix = time.strftime("%m%d_%H%M")  # e.g., 0712_1624
    args.exp = f'TU_InSAR7_{args.vit_name}_{args.img_size}_{time_suffix}'
    snapshot_path = os.path.join("model", args.exp)

    os.makedirs(snapshot_path, exist_ok=True)

    # start training
    trainer_synapse(args, net, snapshot_path)
