import glob, os, re
from PIL import Image
import torch
import numpy as np
from torch.utils.data.dataset import Dataset


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

class ImageDataset(Dataset):
    def __init__(self, root, data_list, transform=None):
        super(ImageDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.filenames = data_list.tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img, target = read_data(os.path.join(self.root, self.filenames[index]))
        img = change_contrast(img, -30)
        if self.transform:
            img = self.transform(img)
        if target.lower() == 'c':
            label = torch.tensor([0, 1])
        else:
            label = torch.tensor([1, 0])
        return img, label


def read_data(fn):
    img = Image.open(fn)
    splitUnderbar = lambda x : re.sub(r'^.+/','',x).replace('.jpg','').split('_')
    target = splitUnderbar(fn)[-1]
    return img, target


def read_files(data_root):
    labels = []
    img_names = os.listdir(data_root)
    for names in img_names:
        splitUnderbar = lambda x : re.sub(r'^.+/','',x).replace('.jpg','').split('_')
        target = splitUnderbar(names)[-1]
        if target.lower() == 'c':
            labels.append(1)
        else:
            labels.append(0)
    return np.array(img_names), np.array(labels)
