from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils, datasets

def get_mnist(params, dset=None):
    dataset = datasets.MNIST(root='./data', download=True,
                         transform=transforms.Compose([
                             transforms.Resize(128),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))


    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=params['bsize'],
                                            shuffle=True)

    return dataloader

def get_other(params, dset=None):
    root = "Dataset/File/Name"
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(root=root,
                                   transform=transform)


    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=params['bsize'],
                                            shuffle=True,drop_last=True)

    return dataloader


