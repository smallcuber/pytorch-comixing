from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch import from_numpy, save, load
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
import json


def convertANIMAL10N():
    dir_test = 'rawdata_ANIMAL10N/testing'
    dir_train = 'rawdata_ANIMAL10N/training'
    _dataset_test = datasets.ImageFolder(dir_test)
    _dataset_train = datasets.ImageFolder(dir_train)


# https://github.com/pxiangwu/PLC/blob/master/animal10/dataset.py
class Animal10N(Dataset):
    def __init__(self, split='train', data_path=None, transform=None):
        if data_path is None:
            data_path = 'rawdata_ANIMAL10N'
        self.mode = split
        self.image_dir = os.path.join(data_path, split + 'ing')
        self.image_files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        self.targets = []

        for path in self.image_files:
            label = path.split('_')[0]
            self.targets.append(int(label))

        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        if self.mode == 'train':
            return image, from_numpy(label), index
        else:
            return image, from_numpy(label)

    def __len__(self):
        return len(self.targets)

    # def update_corrupted_label(self, noise_label):
    #     self.targets[:] = noise_label[:]


if __name__ == '__main__':
    _dir = 'rawdata_ANIMAL10N'
    ds_test = Animal10N(split='test')
    loader_test = DataLoader(dataset=ds_test, batch_size=128, shuffle=True, num_workers=6)

    image, label, index = loader_test
