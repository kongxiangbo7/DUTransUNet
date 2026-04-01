import numpy as np
import torch
import torch.nn as nn
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
import torch.nn.functional as F

# ---------------------------
#          Losses
# ---------------------------
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        修改点：
        - 跳过背景（类0）参与损失与平均，避免稀释前景梯度。
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1.0] * self.n_classes

        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape do not match'

        # 跳过背景类
        valid_classes = list(range(1, self.n_classes))
        if len(valid_classes) == 0:
            return torch.tensor(0.0, device=inputs.device)

        loss = 0.0
        for i in valid_classes:
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / max(1, len(valid_classes))


class IoULoss(nn.Module):
    def __init__(self, n_classes):
        super(IoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _iou_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        union = torch.sum(score) + torch.sum(target) - intersect
        loss = 1 - (intersect + smooth) / (union + smooth)
        return loss

    def forward(self, inputs, target, softmax=False):
        """
        修改点：
        - 跳过背景（类0）参与损失与平均。
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        # 跳过背景类
        valid_classes = list(range(1, self.n_classes))
        if len(valid_classes) == 0:
            return torch.tensor(0.0, device=inputs.device)

        loss = 0.0
        for i in valid_classes:
            iou = self._iou_loss(inputs[:, i], target[:, i])
            loss += iou
        return loss / max(1, len(valid_classes))


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------------------
#     Metrics (per-case)
# ---------------------------
def calculate_metric_percase(pred, gt):
    """
    传入完整二值图（bool/0-1），不再裁剪掉 gt==0 的区域。
    统一约定：
    - gt==0 且 pred==0：Dice=1, IoU=1, HD95=0（空场景下完美一致）
    - gt==0 且 pred>0：Dice=0, IoU=0, HD95=inf（完全失败）
    - gt>0 且 pred==0：Dice=0, IoU=0, HD95=inf（完全失败）
    - 两者均有正例：正常计算
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_pos = int(pred.sum())
    gt_pos = int(gt.sum())

    if gt_pos == 0 and pred_pos == 0:
        return 1.0, 0.0, 1.0
    if gt_pos == 0 and pred_pos > 0:
        return 0.0, np.inf, 0.0
    if gt_pos > 0 and pred_pos == 0:
        return 0.0, np.inf, 0.0

    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    iou = metric.binary.jc(pred, gt)
    return float(dice), float(hd95), float(iou)


def calculate_precision_recall_f1(pred, gt):
    """
    与 calculate_metric_percase 的空集约定保持一致：
    - gt==0 且 pred==0：P=R=F1=1
    - gt==0 且 pred>0：P=R=F1=0
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()

    if TP == 0 and FP == 0 and FN == 0:
        return 1.0, 1.0, 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0, 0.0, 0.0

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1_score)


# ---------------------------
#      Single-volume test
# ---------------------------
def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1):
    image = image[0].cpu().detach().numpy()  # [3, H, W]
    label = label[0].cpu().detach().numpy()  # [H, W]

    # 推理
    input_tensor = torch.from_numpy(image).unsqueeze(0).float().cuda()  # [1, 3, H, W]
    net.eval()
    with torch.no_grad():
        output = net(input_tensor)
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()  # [H, W]
        prediction = out

    # 计算各类指标（跳过背景类 0）
    metric_list = []
    for i in range(1, classes):
        dice, hd95, iou = calculate_metric_percase(prediction == i, label == i)
        precision, recall, f1 = calculate_precision_recall_f1(prediction == i, label == i)
        metric_list.append((dice, hd95, iou, precision, recall, f1))

    # 可视化（拼接一张图）
    import matplotlib.pyplot as plt
    import os

    img = image.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Input")
    axs[1].imshow(prediction, cmap='tab20')
    axs[1].set_title("Prediction")
    axs[2].imshow(label, cmap='tab20')
    axs[2].set_title("Ground Truth")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    if test_save_path is not None and case is not None:
        os.makedirs(test_save_path, exist_ok=True)
        plt.savefig(os.path.join(test_save_path, case + "_compare.png"))
    plt.close()

    return metric_list
