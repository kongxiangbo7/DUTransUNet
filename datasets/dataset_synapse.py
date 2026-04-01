import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def add_gaussian_noise(image, mean=0.0, std=0.1):
    """
    为图像添加高斯噪声
    :param image: 输入图像
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    :return: 添加噪声后的图像
    """
    noise = np.random.normal(mean, std, image.shape)  # 生成高斯噪声
    noisy_image = np.clip(image + noise, 0.0, 1.0)  # 添加噪声并保证图像值在[0, 1]范围内
    return noisy_image


def add_gaussian_blur(image, sigma=1.0):
    """
    为图像添加高斯模糊
    :param image: 输入图像
    :param sigma: 高斯模糊的标准差
    :return: 添加模糊后的图像
    """
    blurred_image = ndimage.gaussian_filter(image, sigma=sigma)
    return blurred_image


def elastic_transform(image, alpha, sigma):
    """
    弹性变换
    :param image: 输入图像
    :param alpha: 变形强度
    :param sigma: 高斯滤波的标准差
    :return: 变形后的图像
    """
    shape = image.shape
    random_state = np.random.RandomState(None)
    dx = ndimage.gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0)
    dy = ndimage.gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0)

    # 为了处理多通道图像，将dx和dy扩展为多通道
    dz = np.zeros_like(dx)  # Assuming 2D image
    dz = ndimage.gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing="xy")
    distored_x = np.clip(x + alpha * dx, 0, shape[1] - 1)
    distored_y = np.clip(y + alpha * dy, 0, shape[0] - 1)

    # 使用多通道图像时，分别对每个通道进行变形
    distorted_image = np.zeros_like(image)
    for i in range(shape[2]):  # 遍历每个颜色通道
        distorted_image[..., i] = ndimage.map_coordinates(image[..., i], [distored_y, distored_x], order=1,
                                                          mode='reflect')

    return distorted_image


class RandomGenerator(object):
    def __init__(self, output_size, noise_std=0.1, blur_sigma=1.0, elastic_alpha=34, elastic_sigma=4.0):
        self.output_size = output_size
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:  # 50%的概率添加噪声
            image = add_gaussian_noise(image, std=self.noise_std)
        if random.random() > 0.5:  # 50%的概率添加模糊
            image = add_gaussian_blur(image, sigma=self.blur_sigma)
        if random.random() > 0.5:  # 50%的概率进行弹性变换
            image = elastic_transform(image, alpha=self.elastic_alpha, sigma=self.elastic_sigma)
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class InSARSegDataset(Dataset):
    def __init__(self, image_root, mask_root, split="train", transform=None, list_txt_root=None):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.split = split

        self.sample_list = []

        # 优先读取列表文件（如 train.txt）
        if list_txt_root:
            list_file = os.path.join(list_txt_root, f"{split}.txt")
            with open(list_file, 'r') as f:
                for line in f:
                    rel_path = line.strip()
                    if not rel_path or rel_path.startswith('#'):  # 忽略空行或注释
                        continue
                    img_path = os.path.join(self.image_root, rel_path)
                    mask_path = os.path.join(self.mask_root, rel_path)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.sample_list.append((img_path, mask_path))
                    else:
                        print(f"[警告] 图像或掩码不存在：{img_path} 或 {mask_path}")
        else:
            # 默认加载所有类别1~6数据
            for i in range(1, 7):
                img_dir = os.path.join(image_root, str(i))
                mask_dir = os.path.join(mask_root, str(i))
                if not os.path.exists(mask_dir): continue
                for filename in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, filename)
                    mask_path = os.path.join(mask_dir, filename)
                    if os.path.exists(mask_path):
                        self.sample_list.append((img_path, mask_path))


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.sample_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = Image.open(mask_path).convert('L')

        image = np.array(image).astype(np.float32) / 255.0
        label = np.array(label).astype(np.int64)
        label[label > 0] = 1

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.basename(img_path)

        sample['mixup'] = False

        return sample
