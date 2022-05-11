import os 
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.transforms import Compose
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
train_label = pd.read_csv("./dataset/train.csv")
test_label = pd.read_csv("./dataset/test.csv")
valid_label = pd.read_csv("./dataset/valid.csv")


class RandomCrop(transforms.RandomCrop):
    def __call__(self, image):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if image.size[0] < self.size[1]:
            p = self.size[1] - image.size[0]
            image = F.pad(image, (p, 0), self.fill, self.padding_mode)
        # pad the height if needed
        if image.size[1] < self.size[0]:
            p = self.size[0] - image.size[1]
            image = F.pad(image, (0, p), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w)


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    data_dir = os.path.join(path,'train')
    T = None
    if use_transform:
        T = Compose([
            RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    dataloader = DataLoader(
        Dataset(data_dir,train_label,transform=T),
        batch_size= batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )
    return dataloader

def test_dataloader(path, batch_size=1, num_workers=0):
    data_dir = os.path.join(path,'test')
    dataloader = DataLoader(
        Dataset(data_dir,test_label),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def valid_dataloader(path, batch_size=1, num_workers=0,use_transform=True):
    data_dir = os.path.join(path,'valid')
    T = None
    if use_transform:
        T = Compose([
            RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    dataloader = DataLoader(
        Dataset(data_dir,valid_label,transform=T),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader

class Dataset(Dataset):
    def __init__(self, data_dir, df, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(data_dir)
        self._check_image(self.image_list)
        self.image_list.sort()
        self.df = df

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_dir, self.image_list[idx]))
        label = self.df.loc[self.df['image'] ==self.image_list[idx]]["class"].iloc[0]
        label = int(label)
        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg','JPG']:
                print(x)
                raise ValueError


