from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import noise_gen



def get_brainwave(params):

    my_dataset = noise_gen.custom_training_set()

    dataset = TensorDataset(my_dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=params['bsize'],
                                             shuffle=True,
                                             drop_last=True)


    return dataloader