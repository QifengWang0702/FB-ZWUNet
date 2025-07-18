from __future__ import print_function, division

import os
import torch
import pandas as pd
from skimage import io, transform
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from random import random, randint
import numpy as np
import cv2
# Ignore warnings
import warnings
import pdb

warnings.filterwarnings("ignore")

CC = [128,128,128]
CV = [128,0,0]
SP = [192,192,128]
CCSP = [128,64,128]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([CC, CV, SP, CCSP])

def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':

        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()
        
        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)
    elif mode == 'val':
        
        train_img_path = os.path.join(root, 'val', 'Img')
        train_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)
            
    else:
        
        train_img_path = os.path.join(root, 'test', 'Img')
        train_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)

    return items

class MedicalImageDataset_org(Dataset):
    """FBUS dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False, flag_multi_class=False, num_class=2):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.flag_multi_class = flag_multi_class
        self.num_class = num_class
    
    def adjustData(self, img, mask):
        if self.flag_multi_class:
            mask_np = np.array(mask)
            new_mask = np.zeros(mask_np.shape + (self.num_class,))
            for i, color in enumerate(COLOR_DICT):
                match = (mask_np == color).all(axis=2)
                new_mask[match, i] = 1
            new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3]))
            mask = to_pil_image(new_mask)
        elif np.max(np.array(img)) > 1:
            img_np = np.array(img) / 255.
            img = Image.fromarray(img_np.astype(np.uint8))
            mask_np = np.array(mask) / 255.
            mask_np[mask_np > 0.5] = 1
            mask_np[mask_np <= 0.5] = 0
            mask = Image.fromarray(mask_np.astype(np.uint8))
        return img, mask
    
    def get_canny_mask(self, mask):
        mask_np = np.array(mask)
        edges = cv2.Canny(mask_np, 0, 1)
        return Image.fromarray(edges)

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)

        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.equalize:
            img = ImageOps.equalize(img)

        # 使用adjustData函数进行预处理
        img, mask = self.adjustData(img, mask)

        # 复制原始的img和mask，用于后续的处理和增强
        original_img = img.copy()
        original_mask = mask.copy()
        
        if self.augmentation:
            img, mask = self.augment(original_img, original_mask)

        mask_canny = self.get_canny_mask(original_mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            mask_canny = self.mask_transform(mask_canny)
        
        return [img, mask, mask_canny, img_path]

class MedicalImageDataset(Dataset):
    """FBUS dataset."""

    def __init__(self, mode, root_dir, transform=None, augment=False, equalize=False):
        """
        Args:
            mode (string): 'train' or 'test' to determine the mode of the dataset.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            augment (bool, optional): Whether to apply augmentation.
            equalize (bool, optional): Whether to apply histogram equalization.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize

    def __len__(self):
        return len(self.imgs)

    def get_canny_mask(self, mask):
        mask_np = np.array(mask)
        edges = cv2.Canny(mask_np, 100, 200)
        return Image.fromarray(edges)

    def augment(self, img, mask, mask_canny):
        # 随机决定是否进行垂直翻转
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
            mask_canny = ImageOps.flip(mask_canny)
        # 随机决定是否进行水平翻转
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
            mask_canny = ImageOps.mirror(mask_canny)
        # 随机决定是否进行旋转
        if random() > 0.5:
            angle = random() * 60 - 30  # 随机角度
            img = img.rotate(angle)
            mask = mask.rotate(angle)
            mask_canny = mask_canny.rotate(angle)

        return img, mask, mask_canny


    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.equalize:
            img = ImageOps.equalize(img)
        
        mask_canny = self.get_canny_mask(mask)
        
        if self.augmentation:
            img, mask, mask_canny = self.augment(img, mask, mask_canny)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            mask_canny = self.transform(mask_canny)
        
        return [img, mask, mask_canny, img_path]

class MedicalImageDataset_week(Dataset):
    """FBUS dataset."""

    def __init__(self, mode, root_dir, info_path, transform=None, augment=False, equalize=False):
        """
        Args:
            mode (string): 'train' or 'test' to determine the mode of the dataset.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            augment (bool, optional): Whether to apply augmentation.
            equalize (bool, optional): Whether to apply histogram equalization.
        """
        self.root_dir = root_dir
        self.week_info = self.load_week_info(info_path)
        self.transform = transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize

    def load_week_info(self, info_path):
        week_info = {}
        with open(info_path, 'r') as file:
            for line in file:
                filename, week = line.strip().split(',')
                if week == 'null':  # 如果周数为null，则输出为0
                    week_info[filename] = 0
                else:
                    # 假设周数是这样的格式 "18w"，去掉最后的"w"并转换为整数
                    week_info[filename] = int(week[:-1])
        return week_info

    def __len__(self):
        return len(self.imgs)

    def get_canny_mask(self, mask):
        mask_np = np.array(mask)
        edges = cv2.Canny(mask_np, 100, 200)
        return Image.fromarray(edges)

    def augment(self, img, mask, mask_canny):
        # 随机决定是否进行垂直翻转
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
            mask_canny = ImageOps.flip(mask_canny)
        # 随机决定是否进行水平翻转
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
            mask_canny = ImageOps.mirror(mask_canny)
        # 随机决定是否进行旋转
        if random() > 0.5:
            angle = random() * 60 - 30  # 随机角度
            img = img.rotate(angle)
            mask = mask.rotate(angle)
            mask_canny = mask_canny.rotate(angle)

        return img, mask, mask_canny


    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img_name = os.path.basename(img_path)
        week_number = self.week_info.get(img_name, 'unknown')
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.equalize:
            img = ImageOps.equalize(img)
        
        mask_canny = self.get_canny_mask(mask)
        
        if self.augmentation:
            img, mask, mask_canny = self.augment(img, mask, mask_canny)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            mask_canny = self.transform(mask_canny)
        
        return [img, mask, mask_canny, img_path, week_number]
