import cv2 
import torch
import numpy as np
from scipy.ndimage.interpolation import map_coordinates


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

        dU = np.random.uniform(-self.ampl, self.ampl, size=self.size)
        dV = np.random.uniform(-self.ampl, self.ampl, size=self.size)

        dU[ 0,:] = 0; dU[-1,:] = 0; dU[:, 0] = 0; dU[:,-1] = 0
        dV[ 0,:] = 0; dV[-1,:] = 0; dV[:, 0] = 0; dV[:,-1] = 0

        dU = cv2.resize(dU, (shape[0], shape[1])) 
        dV = cv2.resize(dV, (shape[0], shape[1])) 
        
        X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(Y+dV, (-1, 1)), np.reshape(X+dU, (-1, 1))
   
        image = map_coordinates(image, indices, order=1).reshape(shape)
        annot = map_coordinates(annot, indices, order=1).reshape(shape)

        return {'image': image, 'annot': annot}


class BoundaryExtension(object):
    
    def __init__(self, ext):
        assert isinstance(ext, int)
        if isinstance(ext, int):
            self.ext = ext

    def __call__(self, sample):
        image, annot = sample['image'], sample['annot']
        
        image = np.pad(image, self.ext, mode='reflect')
        annot = np.pad(annot, self.ext, mode='reflect')

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