from torch.utils.data import  Dataset
import torch
import numpy as np
class Full_Dataset(Dataset):
    def __init__(self,spec,mask,RGB,weights,RGB_balance):
        self.spec = spec
        self.mask = mask
        self.RGB = RGB
        self.weights = weights
        self.RGB_balance = RGB_balance


    def __getitem__(self,index):
        x = self.spec[index]
        y = self.mask[index]
        z = self.RGB[index]
        w = self.weights[index]
        u = self.RGB_balance[index]

        return x, y, z, w, u

    def __len__(self):
        return len(self.spec)


class spec_Dataset(Dataset):
    def __init__(self,spec,mask):
        self.spec = spec
        self.mask = mask
    def __getitem__(self,index):
        k = np.random.randint(4)
        x = torch.rot90(input=self.spec[index],k=k,dims=[0,1])
        y = torch.rot90(input=self.mask[index],k=k,dims=[0,1])
        # x =  self.spec[index]
        # y = self.mask[index]
        return x, y

    def __len__(self):
        return len(self.spec)

class RGB_Dataset(Dataset):
    def __init__(self,RGB,mask):
        self.RGB = RGB
        self.mask = mask
    def __getitem__(self,index):
        k = np.random.randint(4)
        x = torch.rot90(input=self.RGB[index],k=k,dims=[0,1])
        y = torch.rot90(input=self.mask[index],k=k,dims=[0,1])
        return x, y

    def __len__(self):
        return len(self.RGB)