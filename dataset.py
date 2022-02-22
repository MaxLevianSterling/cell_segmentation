from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import *
import cv2   
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates


class LIVECell(Dataset):
    def __init__(self, data_folder, data_subset, transform=None):
        image_folder = f'{data_folder}images/{data_subset}/variables/'
        annot_folder = f'{data_folder}annotations/{data_subset}/variables/'

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
  

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, annot = sample['image'], sample['annot']
        new_h, new_w = self.output_size

        true_points = np.argwhere(image)
        bottom_right = true_points.max(axis=0)

        if bottom_right[0] - new_h > -1:
            top = np.random.randint(0, bottom_right[0] + 1 - new_h)
        else: top = 0
        if bottom_right[1] - new_w > -1:
            left = np.random.randint(0, bottom_right[1] + 1 - new_w)
        else: left = 0

        image = image[top: top + new_h,
                      left: left + new_w]
        annot = annot[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'annot': annot}


class RandomOrientation(object):
    
    def __call__(self, sample):
        image, annot = sample['image'], sample['annot']
        mirror = np.random.randint(1, 5)
        n_rotations = np.random.randint(0, 4)
        
        if mirror > 2:
            image, annot = np.flip(image, 0), np.flip(annot, 0)
        if mirror % 2 == 0:
            image, annot = np.flip(image, 1), np.flip(annot, 1)

        for rotation in range(n_rotations):
            image, annot = np.rot90(image), np.rot90(annot)

        return {'image': image, 'annot': annot}


class LocalDeform(object):

    def __init__(self, size, ampl):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

        assert isinstance(ampl, int)
        self.ampl = ampl

    def __call__(self, sample):
        image, annot = sample['image'], sample['annot']
        shape = image.shape

        du = np.random.uniform(-self.ampl, self.ampl, size=self.size)
        dv = np.random.uniform(-self.ampl, self.ampl, size=self.size)

        du[ 0,:] = 0; du[-1,:] = 0; du[:, 0] = 0; du[:,-1] = 0
        dv[ 0,:] = 0; dv[-1,:] = 0; dv[:, 0] = 0; dv[:,-1] = 0

        # plt.quiver(du, dv)
        # plt.axis('off')
        # plt.show()
        # plt.clf()

        DU = cv2.resize(du, (shape[0], shape[1])) 
        DV = cv2.resize(dv, (shape[0], shape[1])) 
        
        X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
   
        image = map_coordinates(image, indices, order=1).reshape(shape)
        annot = map_coordinates(annot, indices, order=1).reshape(shape)
        
        # Show image
        # plt.imshow(np.hstack( (np.squeeze(image), 
        #                     np.squeeze(deformed_image), 
        #                     np.squeeze(image-deformed_image)
        #                     ), 
        #                     ), cmap = plt.get_cmap('gray'))
        # plt.axis('off')
        # plt.show()

        # Show label
        # plt.imshow(np.hstack( (np.squeeze(annot), 
        #                     np.squeeze(deformed_annot), 
        #                     np.squeeze(annot-deformed_annot)
        #                     ), 
        #                     ), cmap = plt.get_cmap('gray'))
        # plt.axis('off')
        # plt.show()

        return {'image': image, 'annot': annot}


class BoundaryExtension(object):
    
    def __init__(self, ext):
        assert isinstance(ext, int)
        if isinstance(ext, int):
            self.ext = ext

    def __call__(self, sample):
        image, annot = sample['image'], sample['annot']
        shape = image.shape

        image_vflip = np.flip(image, 0)
        image_hflip = np.flip(image, 1)
        image_vhflip = np.flip(np.flip(image, 1),0)

        annot_vflip = np.flip(annot, 0)
        annot_hflip = np.flip(annot, 1)
        annot_vhflip = np.flip(np.flip(annot, 1),0)

        image_left = np.vstack((image_vhflip,image_hflip,image_vhflip))
        image_middle = np.vstack((image_vflip,image,image_vflip))
        image_right = np.vstack((image_vhflip,image_hflip,image_vhflip))

        annot_left = np.vstack((annot_vhflip,annot_hflip,annot_vhflip))
        annot_middle = np.vstack((annot_vflip,annot,annot_vflip))
        annot_right = np.vstack((annot_vhflip,annot_hflip,annot_vhflip))

        image = np.hstack((image_left,image_middle,image_right))
        annot = np.hstack((annot_left,annot_middle,annot_right))

        image = image[shape[0]-self.ext : 2*shape[0]+self.ext,
                      shape[1]-self.ext : 2*shape[1]+self.ext] 
        annot = annot[shape[0]-self.ext : 2*shape[0]+self.ext,
                      shape[1]-self.ext : 2*shape[1]+self.ext] 

        return {'image': image, 'annot': annot}


class Normalize(object):
    
    def __call__(self, sample):
        image, annot = sample['image'], sample['annot']

        image = image.astype(float) / 255
        annot = annot.astype(float) / 255

        return {'image': image, 'annot': annot}
        

class Noise(object):
    
    def __init__(self, std):
        assert std > 0
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        image_shape = image.shape

        image = image + np.random.normal(scale=self.std, size=image_shape) 
        image = np.clip(image, 0, 1)

        return {'image': image, 'annot': sample['annot']}


class ToTensor(object):

    def __call__(self, sample):
        image, annot = sample['image'], sample['annot']

        image = np.expand_dims(image, axis=0)
        annot = np.expand_dims(annot, axis=0)
        
        return {'image': torch.from_numpy(image),
                'annot': torch.from_numpy(annot)}
