import torch
import numpy as np
from torch.utils.data import Dataset
from utils import path_gen
 

class LIVECell(Dataset):
    def __init__(self, path, data_set, data_subset, transform=None):
        image_folder = path_gen([
            path, 
            'data', 
            data_set, 
            'images', 
            data_subset, 
            'variables'
        ])
        annot_folder = path_gen([
            path, 
            'data', 
            data_set, 
            'annotations', 
            data_subset, 
            'variables'
        ])

        self.image_arr = np.load(f'{image_folder}array.npy')
        self.annot_arr = np.load(f'{annot_folder}array.npy')

        self.image_filenames, self.annot_filenames = [], []
        with open(f'{image_folder}filenames.txt', 'r') as infile:
            for line in infile:
                args = line.split('\n')
                self.image_filenames.append(args[0])
        with open(f'{annot_folder}filenames.txt', 'r') as infile:
            for line in infile:
                args = line.split('\n')
                self.annot_filenames.append(args[0])

        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)
   
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        identifier = self.image_filenames[idx]
        annot_idx = self.annot_filenames.index(identifier)
        sample = {'image': self.image_arr[idx, :, :], 
                  'annot': self.annot_arr[annot_idx, :, :]}

        if self.transform:
            sample = self.transform(sample)

        return sample