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
        values = [value for value in sample.values()]
        new_h, new_w = self.output_size

        true_points = np.argwhere(values[0])
        bottom_right = true_points.max(axis=0)

        if bottom_right[0] - new_h > -1:
            top = np.random.randint(0, bottom_right[0] + 1 - new_h)
        else: top = 0
        if bottom_right[1] - new_w > -1:
            left = np.random.randint(0, bottom_right[1] + 1 - new_w)
        else: left = 0

        for iV in range(len(values)):
            values[iV] = values[iV][
                top: top + new_h,
                left: left + new_w
            ]

        return {list(sample.keys())[iV]: values[iV] for iV in range(len(values))}


class RandomOrientation(object):
    
    def __call__(self, sample):
        values = [value for value in sample.values()]
        mirror = np.random.randint(1, 5)
        n_rotations = np.random.randint(0, 4)
        
        if mirror > 2:
            values = [np.flip(values[iV], 0) for iV in range(len(values))]
        if mirror % 2 == 0:
            values = [np.flip(values[iV], 1) for iV in range(len(values))]

        for _ in range(n_rotations):
            values = [np.rot90(values[iV]) for iV in range(len(values))]

        return {list(sample.keys())[iV]: values[iV] for iV in range(len(values))}


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
        values = [value for value in sample.values()]
        shape = values[0].shape

        dS = [np.random.uniform(-self.ampl, self.ampl, size=self.size) for iS in range(2)]

        for iS in range(2):
            for n in [0, -1]:
                dS[iS][n, :] = 0
                dS[iS][:, n] = 0

        dS = [cv2.resize(dS[iS], (shape[0], shape[1])) for iS in range(2)]       
        
        X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(Y+dS[1], (-1, 1)), np.reshape(X+dS[0], (-1, 1))
    
        values = [map_coordinates(values[iV], indices, order=1).reshape(shape) for iV in range(len(values))]

        return {list(sample.keys())[iV]: values[iV] for iV in range(len(values))}


class BoundaryExtension(object):
    
    def __init__(self, ext):
        assert isinstance(ext, int)
        if isinstance(ext, int):
            self.ext = ext

    def __call__(self, sample):
        values = [value for value in sample.values()]
        
        values = [np.pad(values[iV], self.ext, mode='reflect') for iV in range(len(values))]

        return {list(sample.keys())[iV]: values[iV] for iV in range(len(values))}


class Normalize(object):
    
    def __call__(self, sample):
        values = [value for value in sample.values()]

        values = [values[iV].astype(float) / 255 for iV in range(len(values))]

        return {list(sample.keys())[iV]: values[iV] for iV in range(len(values))}
        

class Noise(object):
    
    def __init__(self, std):
        assert std > 0
        self.std = std

    def __call__(self, sample):
        values = [value for value in sample.values()]

        values[0] += np.random.normal(scale=self.std, size=values[0].shape) 
        values[0] = np.clip(values[0], 0, 1)

        return {list(sample.keys())[iV]: values[iV] for iV in range(len(values))}


class ToTensor(object):

    def __call__(self, sample):
        values = [value for value in sample.values()]

        values = [np.expand_dims(values[iV], axis=0) for iV in range(len(values))]

        return {list(sample.keys())[iV]: torch.from_numpy(values[iV]) for iV in range(len(values))}