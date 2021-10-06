from __future__ import absolute_import
import torch
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch.nn as nn

__all__ = ['ContrastiveDataset', 'ImageDataset', 'TripleDataset', "UDataset", "RawDataset"]

########################## Contrastive Learning ##########################
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class ContrastiveDataset(Dataset):
    """Construct A Constrastive Dataset"""
    @staticmethod
    def get_simclr_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])
        return data_transforms

    def __init__(self, X, n_views=2, size=32):
        self.X = X
        self.CL_transform = ContrastiveLearningViewGenerator(
            base_transform=self.get_simclr_transform(size), 
            n_views=n_views
        )

    def __getitem__(self, index):
        img = self.X[index]
        img = Image.fromarray(img)
        img = self.CL_transform(img)
        return img

    def __len__(self): return len(self.X)

class RawDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

    def __len__(self): return len(self.X)

class ImageDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        if transform is None:
            self.transform = transforms.Compose([ 
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img = self.X[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, self.y[index]

    def __len__(self): return len(self.X)

class UDataset(Dataset):
    def __init__(self, X):
        self.X = X
        self.transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])

    def __getitem__(self, index):
        img = self.X[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img

    def __len__(self): return len(self.X)

class TripleDataset(Dataset):
    def __init__(self, X, I, D):
        self.n = len(X)
        self.X = X
        self.I = I
        self.D = D
        self.pointer = [0 for i in range(X.shape[0])]
        self.random = 0
        self.K = self.D.shape[1]
        self.SEED = 1e9 + 7
        

    def __getitem__(self, index):
        self.pointer[index] += 1
        if self.pointer[index] >= self.K: self.pointer[index] = 1
        self.random = int((self.random + self.SEED) % self.n)

        imgA = self.X[index, :]
        imgB = self.X[self.I[index, self.pointer[index]], :]
        imgC = self.X[self.random, :]

        return imgA, imgB, imgC, self.D[index, self.pointer[index]]

    def __len__(self): return len(self.X)

