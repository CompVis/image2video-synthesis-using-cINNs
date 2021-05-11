import cv2, torch, torch.nn as nn
import numpy as np
import kornia as k, os, glob


class Augmentation_random_crop(nn.Module):
    def __init__(self, img_size, params):
        super(Augmentation_random_crop, self).__init__()
        self.img_size = img_size
        self.resize = k.Resize(size=(self.img_size + 16, self.img_size + 16))
        self.random_crop = k.augmentation.RandomCrop(size=(self.img_size, self.img_size), same_on_batch=True)
        self.jit = k.augmentation.ColorJitter(brightness=params['brightness'], contrast=params['contrast'],
                                              saturation=params['saturation'], hue=params['hue'], same_on_batch=True)
        self.hflip = k.augmentation.RandomHorizontalFlip(same_on_batch=True, p=params['prob_hflip'])
        self.normalize = k.augmentation.Normalize(0.5, 0.5)


    def forward(self, x):
        x = self.resize(x)
        x = self.hflip(x)
        x = self.random_crop(x)
        x = self.jit(x)
        return self.normalize(x)


class Augmentation(nn.Module):
    def __init__(self, img_size, params):
        super(Augmentation, self).__init__()
        self.img_size = img_size
        self.resize = k.Resize(size=(self.img_size, self.img_size))
        self.jit = k.augmentation.ColorJitter(brightness=params['brightness'], contrast=params['contrast'],
                                              saturation=params['saturation'], hue=params['hue'], same_on_batch=True)
        self.normalize = k.augmentation.Normalize(0.5, 0.5)
        self.hflip = k.augmentation.RandomHorizontalFlip(same_on_batch=True, p=params['prob_hflip'])

    def forward(self, x):
        x = self.resize(x)
        x = self.hflip(x)
        x = self.jit(x)
        return self.normalize(x)