import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from config import IMAGE_SIZE


class PolypDataset(Dataset):
    """Polyp segmentation dataset with optional augmentation."""

    def __init__(self, img_path, mask_path, augment=False):

        self.img_path = img_path
        self.mask_path = mask_path
        self.augment_flag = augment

        self.images = sorted(os.listdir(img_path))

    def __len__(self):
        return len(self.images)

    def _augment(self, img, mask):

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        if np.random.rand() < 0.5:
            angle = np.random.randint(-30, 30)
            center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE))
            mask = cv2.warpAffine(mask, M, (IMAGE_SIZE, IMAGE_SIZE))

        return img, mask

    def __getitem__(self, idx):

        name = self.images[idx]

        img = cv2.imread(os.path.join(self.img_path, name))
        mask = cv2.imread(os.path.join(self.mask_path, name), 0)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        if self.augment_flag:
            img, mask = self._augment(img, mask)

        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask
