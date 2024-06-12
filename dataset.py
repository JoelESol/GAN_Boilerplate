import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as trans

## Import Data Loaders ##
from dataloader import *


def get_dataset(dataset, batch, imsize):
    if dataset == 'SC':
        train_dataset = Synth_Dataset(root='./data/Synthetic_Can_Data', train=True,
                                     transform=transforms.Compose([
                                         trans.Resize(imsize),
                                         trans.ToTensor(),
                                         trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
        test_dataset = Synth_Dataset(root='./data/Synthetic_Can_Data', train=False,
                                      transform=transforms.Compose([
                                          trans.Resize(imsize),
                                          trans.ToTensor(),
                                          trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))

    elif dataset == 'PC':
        train_dataset = Phys_Dataset(root='./data/physical_can', train=True,
                                     transform=transforms.Compose([
                                         trans.Resize(imsize),
                                         trans.ToTensor(),
                                         trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
        test_dataset = Phys_Dataset(root='./data/physical_can', train=False,
                                     transform=transforms.Compose([
                                         trans.Resize(imsize),
                                         trans.ToTensor(),
                                         trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
    elif dataset == 'SC_gen':
        train_dataset = Generated_Dataset(root='./data/generated_can_data', train=True,
                                          transform=transforms.Compose([
                                              trans.Resize(imsize),
                                              trans.ToTensor(),
                                              trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ]))
        test_dataset = Phys_Dataset(root='./data/generated_can_data', train=False,
                                    transform=transforms.Compose([
                                        trans.Resize(imsize),
                                        trans.ToTensor(),
                                        trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch*4, shuffle=False)
    return train_dataloader, test_dataloader
