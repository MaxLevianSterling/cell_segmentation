import cv2 
import torch
import numpy                        as np
from math                           import ceil
from utils                          import unpad
from scipy.ndimage.interpolation    import map_coordinates


class RandomCrop(object):
    """Randomly crops a 2D numpy array"""

    def __init__(self, input_size, output_size):
        """ Args:
            input_size (int/tuple): input image sizes
            output_size (int/tuple): output image sizes
        """      

        assert isinstance(input_size, (int, tuple))
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size
            
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """ Args:
            sample (string:np.array dict): input images
                    
        Returns:
            (string:np.array dict): output images
        """    

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Randomly find cropping position
        if self.input_size[0] - self.output_size[0] > -1:
            top = np.random.randint(
                0, 
                self.input_size[0] + 1 - self.output_size[0]
            )
        else: top = 0
        if self.input_size[1] - self.output_size[1] > -1:
            left = np.random.randint(
                0, 
                self.input_size[1] + 1 - self.output_size[1]
            )
        else: left = 0

        # Crop
        for iV in range(len(values)):
            values[iV] = values[iV][
                top: top + self.output_size[0],
                left: left + self.output_size[1]
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class RandomOrientation(object):
    """Randomly orients a 2D numpy array into one of
    eight orientations (all 90 degree rotations and 
    mirrors)
    """

    def __call__(self, sample):
        """ Args:
            sample (string:np.array dict): input images
                    
        Returns:
            (string:np.array dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Randomly select orientation
        mirror = np.random.randint(1, 5)
        n_rotations = np.random.randint(0, 4)
        
        # Vertical flip
        values = [
            np.flip(values[iV], 0) 
            if mirror > 2 
            else values[iV] 
            for iV in range(len(values))
        ]

        # Horizontal flip
        values = [
            np.flip(values[iV], 1) 
            if mirror % 2 == 0 
            else values[iV] 
            for iV in range(len(values))
        ]

        # Counterclockwise rotation
        values = [
            np.rot90(values[iV], n_rotations) 
            for iV in range(len(values))
        ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class LocalDeform(object):
    """Locally deforms a 2D numpy array based on
    a randomly generated sparse vector array
    """

    def __init__(self, size, ampl):
        """ Args:
            size (int/tuple): number of deforming \
                vectors along each axis
            ampl (int): maximum vector magnitude
        """    

        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

        assert isinstance(ampl, int)
        self.ampl = ampl

    def __call__(self, sample):
        """ Args:
            sample (string:np.array dict): input images
                    
        Returns:
            (string:np.array dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Get input shape
        shape = values[0].shape

        # Initialize random sparse vector field
        dS = [
            np.random.uniform(-self.ampl, self.ampl, size=self.size) 
            for iS in range(2)
        ]

        # Zero out the edges
        for iS in range(2):
            for n in [0, -1]:
                dS[iS][n, :] = 0
                dS[iS][:, n] = 0

        # Resize vector field to pixel resolution
        dS = [
            cv2.resize(dS[iS], (shape[0], shape[1])) 
            for iS in range(2)
        ]       
        
        # Determine axis-wise pixel transformations
        X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(Y+dS[1], (-1, 1)), np.reshape(X+dS[0], (-1, 1))

        # Deform
        values = [
            map_coordinates(values[iV], indices, order=1).reshape(shape) 
            for iV in range(len(values))
        ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class Padding(object):
    """Pads a numpy array or list of numpy arrays 
    with reflected values
    """

    def __init__(self, width):
        """ Args:
            width (int): Padding width
        """ 

        assert isinstance(width, int)
        if isinstance(width, int):
            self.width = width

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:[np.array] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Pad
        if isinstance(values[0], list):
            values = [
                [
                    np.pad(
                        values[iV][iC], 
                        (
                            (0, 0), 
                            (self.width, self.width), 
                            (self.width, self.width)
                        ), 
                        mode='reflect'
                    ) 
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        else:
            values = [
                np.pad(values[iV], self.width, mode='reflect') 
                for iV in range(len(values))
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToUnitInterval(object):
    """Converts a numpy array or list of numpy arrays 
    from uint8 range to the unit interval
    """

    def __call__(self, sample):
        """ Args:
            sample (string:np.array dict): input images
                    
        Returns:
            (string:np.array.float dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Transform to unit interval
        if isinstance(values[0], list):
            values = [
                [
                    values[iV][iC].astype(float) / 255
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        else:
            values = [
               values[iV].astype(float) / 255 
               for iV in range(len(values))
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToBinary(object):
    """Converts a numpy array or list of numpy arrays 
    from unit interval range to binary based on a cutoff
    """

    def __init__(self, cutoff, items):
        """ Args:
            cutoff (float): float to binary cutoff point
            items (list): list of sample value numbers 
                to convert to binary
        """   

        assert isinstance(cutoff, float)
        assert cutoff >= 0 and cutoff < 1
        self.cutoff = cutoff
        
        assert isinstance(items, list)
        self.items = items

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array].float dict): input images
                    
        Returns:
            (string:np.array.uint8 dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Transform to binary as per cutoff
        if isinstance(values[0], list):
            values = [
                [
                    np.where(values[iV][iC] > self.cutoff, 1, 0)
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values)) 
                if iV in self.items
            ]            
        else:
            values = [
                np.where(values[iV] > self.cutoff, 1, 0).astype('uint8')
                if iV in self.items
                else values[iV]
                for iV in range(len(values))
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class Noise(object):
    """Adds noise to a numpy array or list of numpy 
    arrays 
    """

    def __init__(self, std, items):
        """ Args:
            std (float): standard deviation of Gaussian
                noise
            items (list): list of sample value numbers 
                to convert to binary
        """  

        assert isinstance(std, float) and std > 0
        self.std = std

        assert isinstance(items, list)
        self.items = items

    def __call__(self, sample):
        """ Args:
            sample (string:np.array dict): input images
                    
        Returns:
            (string:np.array dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Add noise and clip at unit interval
        values = [
            np.clip(
                values[0] + np.random.normal(
                    scale=self.std, 
                    size=values[0].shape
                ), 0, 1
            ) 
            if iV in self.items
            else values[iV]
            for iV in range(len(values))
        ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToTensor(object):
    """Converts a numpy array or list of numpy arrays to
    PyTorch tensor format
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:tensor dict): output tensors
        """   

        # Get input dictionary values        
        values = [
            value 
            for value in sample.values()
        ]

        # Convert to tensor and expand 1st dimension
        if isinstance(values[0], list):
            values = [
                [
                    np.expand_dims(values[iV][iC], axis=1)
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]            
            values = [
                [
                    torch.from_numpy(values[iV][iC])
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        else:
            values = [
                np.expand_dims(values[iV], axis=0) 
                for iV in range(len(values))
            ]
            values = [
                torch.from_numpy(values[iV]) 
                for iV in range(len(values))
            ]
            
        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class FullCrop(object):
    """Crops a numpy array regularly to get a list of crops
    comprising the entire array
    """

    def __init__(self, input_size, output_size):
        """ Args:
            input_size (int/tuple): input image sizes
            output_size (int/tuple): output image size
        """    

        assert isinstance(input_size, (int, tuple))
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """ Args:
            sample (string:np.array dict): input images
                    
        Returns:
            (string:[np.arrays] dict): output images
        """   

        # Get input dictionary values        
        values = [
            value 
            for value in sample.values()
        ]

        # Determine how many crops needed for each axis
        n_crops_h = ceil(self.input_size[0] / self.output_size[0])
        n_crops_w = ceil(self.input_size[1] / self.output_size[1])

        # Determine the cropping positions
        tops = [
            iCh*self.output_size[0] 
            for iCh in range(n_crops_h-1)
        ]
        tops.append(self.input_size[0]-self.output_size[0])
        lefts = [
            iCw*self.output_size[1] 
            for iCw in range(n_crops_w-1)
        ]
        lefts.append(self.input_size[1]-self.output_size[1])
       
        # Crop
        cropped_samples = [[] for value in values]
        for iV in range(len(values)):
            for top in tops:
                for left in lefts:
                    cropped_samples[iV].append(values[iV][
                        top: top + self.output_size[0],
                        left: left + self.output_size[1]
                    ])

        return {
            list(sample.keys())[iV]: cropped_samples[iV] 
            for iV in range(len(values))
        }


class StackOrient(object):
    """Creates a stack of all 8 numpy array orientations   
    attainable by a combination of 90 degree rotations 
    in place of each numpy array input list item
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:[np.arrays] dict): output images
        """   

        # Get input dictionary values        
        values = [
            value 
            for value in sample.values()
        ]
        
        # Stack and orient 
        for iV in range(len(values)):
            for iC in range(len(values[0])):

                mirrored_stack = np.stack((
                    values[iV][iC],
                    np.flip(values[iV][iC], 0),
                    np.flip(values[iV][iC], 1),
                    np.flip(values[iV][iC], (0, 1))
                ), axis = 0)
                
                values[iV][iC] = np.concatenate((
                    mirrored_stack,
                    np.rot90(mirrored_stack, axes=(1, 2))
                ))

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class Squeeze(object):
    """Squeezes a numpy array or list of numpy arrays 
    in the 1st dimension
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:[np.array] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Squeeze
        if isinstance(values[0], list):
            values = [
                [
                    np.squeeze(values[iV][iC], axis=1)
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        else:
            values = [
                np.squeeze(values[iV], axis=1) 
                for iV in range(len(values))
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToFullInterval(object):
    """Converts a numpy array or list of numpy arrays 
    from the unit interval to uint8 range
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:[np.array] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Convert to full uint8 range
        if isinstance(values[0], list):
            values = [
                [
                    values[iV][iC] * 255
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        else:
           values = [
               values[iV] * 255 
               for iV in range(len(values))
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class Unpadding(object):
    """Unpads a numpy array or list of numpy arrays"""

    def __init__(self, width):
        """ Args:
            width (int): unpadding width
        """   

        assert isinstance(width, int)
        if isinstance(width, int):
            self.width = width

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:[np.array] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Unpad
        if isinstance(values[0], list):
            values = [
                [
                    unpad(
                        values[iV][iC], 
                        (
                            (0, 0), 
                            (self.width, self.width), 
                            (self.width, self.width)
                        )
                    ) 
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        else:
            values = [
                unpad(
                    values[iV], 
                    (
                        (0, 0), 
                        (self.width, self.width), 
                        (self.width, self.width)
                    )
                ) 
                for iV in range(len(values))
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class StackReorient(object):
    """Reorients a list of numpy array stack of all 8 
    orientations attainable by a combination of 90 degree 
    rotations back into their original position
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:[np.array] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]
        
        # Reorient all stack layers
        for iV in range(len(values)):
            for iC in range(len(values[0])):
                values[iV][iC][4:, :, :] = np.rot90(
                    values[iV][iC][4:, :, :], 
                    3, 
                    axes=(1, 2)
                )
                values[iV][iC] = np.stack((
                    values[iV][iC][0, :, :],
                    np.flip(values[iV][iC][1, :, :], 0),
                    np.flip(values[iV][iC][2, :, :], 1),
                    np.flip(values[iV][iC][3, :, :], (0, 1)),
                    values[iV][iC][4, :, :],
                    np.flip(values[iV][iC][5, :, :], 0),
                    np.flip(values[iV][iC][6, :, :], 1),
                    np.flip(values[iV][iC][7, :, :], (0, 1))
                ), axis = 0)

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class StackMean(object):
    """Averages a list of numpy array stacks across the 0th 
    dimension
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:[np.array] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]
        
        # Average
        values = [
            [
                np.mean(values[iV][iC], axis=0)
                for iC in range(len(values[0]))
            ]
            for iV in range(len(values))
        ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }
        
        
class Uncrop(object):
    """Merges a list of regularly cropped numpy arrays
    back into an averaged array in the shape of the
    original array
    """

    def __init__(self, input_size, output_size):
        """ Args:
            input_size (int/tuple): input image sizes
            output_size (int/tuple): output image size
        """  
         
        assert isinstance(input_size, (int, tuple))
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size
            
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """ Args:
            sample (string:[np.array] dict): input images
                    
        Returns:
            (string:np.array dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Determine how many crops needed for each axis
        n_crops_h = ceil(self.output_size[0] / self.input_size[0])
        n_crops_w = ceil(self.output_size[1] / self.input_size[1])

        # Determine cropping positions
        tops = [
            iCh*self.input_size[0] 
            for iCh in range(n_crops_h-1)
        ]
        tops.append(self.output_size[0]-self.input_size[0])
        lefts = [
            iCw*self.input_size[1] 
            for iCw in range(n_crops_w-1)
        ]
        lefts.append(self.output_size[1]-self.input_size[1])
       
        # Initialize output array
        prediction = [
            np.repeat(
                np.zeros((self.output_size), dtype=float), 
                n_crops_h * n_crops_w, 
                axis=0
            )
            for iV in range(len(values))
        ]

        # Pad crops with zeros
        for iV in range(len(values)):
            for iT, top in enumerate(tops):
                for iL, left in enumerate(lefts):
                    prediction[iV][
                        2*iT + iL,
                        top: top + self.input_size[0],
                        left: left + self.input_size[1]
                    ] = values[iV][2*iT + iL] 
        
        # Transform zero to NaN
        prediction = np.where(
            prediction == 0,
            np.nan,
            prediction
        )

        # Merge
        prediction = [
            np.nanmean(prediction, axis=0)
            for iV in range(len(values))
        ]

        return {
            list(sample.keys())[iV]: prediction[iV] 
            for iV in range(len(values))
        }