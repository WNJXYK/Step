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


__all__ = ['CIFAR10Mix', 'CIFAR100Mix', 'CIFAR10Lim', 'CIFAR100Lim', 'load_data']
DATASET_ROOT = './data/'
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d): continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images

def load_data(in_dataset="Cifar10", out_dataset="Imagenet", n_labels=250, n_validation=1000, seed=0, u_validation=50):
    '''Load L, UL, Valid, Test data from in-distribution & out-of-distribution dataset'''
    train, test = {}, {}
    # Read Dataset
    if in_dataset == 'Cifar10': 
        iTr = datasets.CIFAR10(DATASET_ROOT,  True,  download=True)
        iTe = datasets.CIFAR10(DATASET_ROOT,  False, download=True)
        
    if in_dataset == 'Cifar100': 
        iTr = datasets.CIFAR100(DATASET_ROOT, True,  download=True)
        iTe = datasets.CIFAR100(DATASET_ROOT, False, download=True)
        num_classes = 100
    oTe = make_dataset(os.path.join(DATASET_ROOT, out_dataset))
    
    # Permutation Data with fixed Te & Va
    np.random.seed(seed)
    iTr_images, iTr_labels = iTr.data, np.array(iTr.targets)
    permut = np.random.permutation(len(iTr_images))
    iTr_images = iTr_images[permut]
    iTr_labels = iTr_labels[permut]

    np.random.seed(0)
    iTe_images, iTe_labels = iTe.data, np.array(iTe.targets)
    permut = np.random.permutation(len(iTe_images))
    iTe_images = iTe_images[permut]
    iTe_labels = iTe_labels[permut]
    np.random.seed(0)
    permut = np.random.permutation(len(oTe))
    oTe_images = np.array([np.asarray(transforms.Resize(32)(pil_loader(oTe[i]))) for i in permut])
    oTe_labels = np.array([-1] * len(oTe))

    # Split Data
    classes = np.unique(iTr_labels)
    n_labels_per_cls = n_labels // len(classes)
    l_images, l_labels = [], []
    u_images, u_labels = [], []
    v_images, v_labels = [], []
    t_images, t_labels = [], []
    for c in classes:
        cls_mask = (iTr_labels == c)
        c_images = iTr_images[cls_mask]
        c_labels = iTr_labels[cls_mask]
        l_images += [c_images[: n_labels_per_cls]]
        l_labels += [c_labels[: n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls: ]]
        u_labels += [c_labels[n_labels_per_cls: ]]
    u_images += [oTe_images[n_validation: ]]
    u_labels += [oTe_labels[n_validation: ]]
    v_images += [iTe_images[: n_validation][: u_validation]] + [oTe_images[: n_validation][: u_validation]]
    v_labels += [iTe_labels[: n_validation][: u_validation]] + [oTe_labels[: n_validation][: u_validation]]
    t_images += [iTe_images[n_validation: ]] + [oTe_images[n_validation: ]]
    t_labels += [iTe_labels[n_validation: ]] + [oTe_labels[n_validation: ]]

    # Return Data
    Tr_L = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    Tr_U = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}
    Va = {"images": np.concatenate(v_images, 0), "labels": np.concatenate(v_labels, 0)}
    Te = {"images": np.concatenate(t_images, 0), "labels": np.concatenate(t_labels, 0)}
    print("ID ", in_dataset, "; OOD", out_dataset)
    print(" > Train Labeled", Tr_L["images"].shape, Tr_L["labels"].shape)
    print(" > Train Unlabeled", Tr_U["images"].shape, Tr_U["labels"].shape)
    print(" > Valid", Va["images"].shape, Va["labels"].shape)
    print(" > Test", Te["images"].shape, Te["labels"].shape)
    
    return Tr_L, Tr_U, Va, Te

# if __name__ == "__main__": load_data()

class CIFAR10Lim(torchvision.datasets.CIFAR10):
    def __init__(self, root, size=0.01, train=False, transform=None, target_transform=None, download=False, SEED=19260817):
        super(CIFAR10Lim, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        np.random.seed(SEED)
        perm = np.random.permutation(len(self.data))
        coll = []
        
        num_classes = 10
        if size < 1: size = int(len(self.data) * size / num_classes)
        size = int(size)
        count = [0 for i in range(num_classes)]
        print("Label Size:", size * num_classes)

        for i in perm.tolist():
            target = self.targets[i]
            if count[target] < size:
                count[target] += 1
                coll.append(i)
        self.data = self.data[coll]
        self.targets = [self.targets[i] for i in coll]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR10Mix(torchvision.datasets.CIFAR10):
    def __init__(self, root, out_path, train=False, val=False, transform=None, target_transform=None, download=False, SEED=19260817):
        super(CIFAR10Mix, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.outpath = make_dataset(out_path)
        if val:
            np.random.seed(SEED)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[:1000]]
            self.targets = [self.targets[i] for i in p1.tolist()[:1000]]
            np.random.seed(SEED)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[:1000]]
        else:
            np.random.seed(SEED)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[1000:]]
            self.targets = [self.targets[i] for i in p1.tolist()[1000:]]
            np.random.seed(SEED)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[1000:]]

    def __getitem__(self, index):
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
        else:
            img_path, target = self.outpath[index - len(self.data)], -1
            img = pil_loader(img_path)
            img = transforms.Resize(32)(img)
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data) + len(self.outpath)

class CIFAR100Lim(torchvision.datasets.CIFAR100):
    def __init__(self, root, size=0.01, train=False, transform=None, target_transform=None, download=False, SEED=19260817):
        super(CIFAR100Lim, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        np.random.seed(SEED)
        perm = np.random.permutation(len(self.data))
        coll = []
        
        num_classes = 100
        if size < 1: size = int(len(self.data) * size / num_classes)
        size = int(size)
        count = [0 for i in range(num_classes)]
        print("Label Size:", size * num_classes)

        for i in perm.tolist():
            target = self.targets[i]
            if count[target] < size:
                count[target] += 1
                coll.append(i)
        self.data = self.data[coll]
        self.targets = [self.targets[i] for i in coll]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR100Mix(torchvision.datasets.CIFAR100):
    def __init__(self, root, out_path, train=False, val=False, transform=None, target_transform=None, download=False, SEED=19260817):
        super(CIFAR100Mix, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.outpath = make_dataset(out_path)
        if val:
            np.random.seed(SEED)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[:1000]]
            self.targets = [self.targets[i] for i in p1.tolist()[:1000]]
            np.random.seed(SEED)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[:1000]]
        else:
            np.random.seed(SEED)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[1000:]]
            self.targets = [self.targets[i] for i in p1.tolist()[1000:]]
            np.random.seed(SEED)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[1000:]]

    def __getitem__(self, index):
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
        else:
            img_path, target = self.outpath[index - len(self.data)], -1
            img = pil_loader(img_path)
            img = transforms.Resize(32)(img)
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data) + len(self.outpath)