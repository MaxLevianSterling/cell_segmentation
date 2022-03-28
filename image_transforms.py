import cv2 
import torch
import numpy                        as np
import torchvision.utils            as v_utils
import torch.nn.functional          as tfunc
from math                           import ceil
from utils                          import unpad
from utils                          import map_tensor_coordinates
from utils                          import HiddenPrints
from scipy.ndimage.interpolation    import map_coordinates
import warnings


class RandomCrop(object):
    """Randomly crops input """

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
            sample (string:(HxW np.array or BxCxHxW tensor) 
                dict): input images
                    
        Returns:
            (string:(HxW np.array or BxCxHxW tensor) dict): 
                output images
        """    

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]    
        
        # Randomly find cropping position
        top = torch.randint(
            self.input_size[0] + 1 - self.output_size[0],
            (1,)
        )
        left = torch.randint(
            self.input_size[1] + 1 - self.output_size[1],
            (1,)
        )

        # Crop
        for iV in range(len(values)):
            if isinstance(values[iV], np.ndarray) and len(values[iV].shape) == 2:
                values[iV] = values[iV][
                    top: top + self.output_size[0],
                    left: left + self.output_size[1]
                ]
            elif torch.is_tensor(values[iV]) and len(values[iV].shape) == 4:
                values[iV] = values[iV][
                    :,
                    :,
                    top: top + self.output_size[0],
                    left: left + self.output_size[1]
                ]
        
        # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_crop_image_snapshot_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_crop_annot_snapshot_image.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class RandomOrientation(object):
    """Randomly orients input into one of eight orientations 
    (all 90 degree rotations and mirrors)
    """

    def __call__(self, sample):
        """ Args:
            sample (string:(HxW np.array or BxCxHxW tensor) 
                dict): input images
                    
        Returns:
            (string:(HxW np.array or BxCxHxW tensor) dict): 
                output images
        """   
        
        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]
            
        # Randomly select orientation
        mirror = torch.randint(low=1, high=5, size=(1,))
        n_rotations = torch.randint(low=0, high=4, size=(1,))   

        for iV in range(len(values)):
            if isinstance(values[iV], np.ndarray) and len(values[iV].shape) == 2:

                # Vertical flip
                values[iV] = np.flip(values[iV], 0) if mirror > 2 else values[iV]

                # Horizontal flip
                values[iV] = np.flip(values[iV], 1) if mirror % 2 == 0 else values[iV]

                # Counterclockwise rotation
                values[iV] = np.rot90(values[iV], n_rotations)

            elif torch.is_tensor(values[iV]) and len(values[iV].shape) == 4:
                
                # Vertical flip
                values[iV] = torch.flip(values[iV], [2]) if mirror > 2 else values[iV]

                # Horizontal flip
                values[iV] = torch.flip(values[iV], [3]) if mirror % 2 == 0 else values[iV]

                # Counterclockwise rotation
                values[iV] = torch.rot90(values[iV], n_rotations.item(), [2, 3])

        # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_orient_image_snapshot_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_orient_annot_snapshot_image.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class LocalDeform(object):
    """Locally deforms input based on a randomly 
    generated sparse vector array
    """

    def __init__(self, size, ampl):
        """ Args:
            size (int/tuple): number of deforming vectors 
                along each axis
            ampl (int): maximum vector magnitude in pixels
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
            sample (string:(HxW np.array or BxCxHxW tensor) 
                dict): input images
                    
        Returns:
            (string:(HxW np.array or BxCxHxW tensor) dict): 
                output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Get input shape
        shape = values[0].shape

        if isinstance(values[0], np.ndarray) and len(shape) == 2:

            # Initialize random sparse vector field
            dS = [
                np.random.uniform(-self.ampl, self.ampl, size=self.size) 
                for iS in range(2)
            ]

            # Zero out edges
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

        elif torch.is_tensor(values[0]) and len(shape) == 4:
            
            # Get current GPU device
            current_device = f'cuda:{values[0].get_device()}'
            
            # Initialize random sparse vector field
            dS = [
                torch.FloatTensor(
                    shape[0], 
                    1, 
                    self.size[0] - 2, 
                    self.size[1] - 2
                ).uniform_(-self.ampl, self.ampl).to(device=current_device, dtype=torch.float)
                for iS in range(2)
            ]

            # Pad the edges with zeros
            dS = [
                tfunc.pad(dS[iS], (1, 1, 1, 1), "constant", 0)
                for iS in range(2)
            ]
 
            # Resize vector field to pixel resolution
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dS = [
                    tfunc.interpolate(
                        dS[iS], 
                        scale_factor=(shape[2]/self.size[0], shape[3]/self.size[1]), 
                        align_corners=True, 
                        mode='bilinear'
                    )
                    for iS in range(2)
                ]       
        
            # Determine axis-wise pixel transformations
            X, Y = torch.meshgrid(torch.arange(shape[2]).to(device=current_device, dtype=torch.float), torch.arange(shape[3]).to(device=current_device, dtype=torch.float), indexing='xy')
            X = X.unsqueeze(0).repeat(shape[0], 1, 1).unsqueeze(1)
            Y = Y.unsqueeze(0).repeat(shape[0], 1, 1).unsqueeze(1)
            indices = torch.reshape(Y+dS[1], (-1, 1)), torch.reshape(X+dS[0], (-1, 1))
            indices = torch.cat((indices[0], indices[1]), dim=1)
            indices = torch.transpose(indices, 0, 1)

            # Create batch and channel indices
            chan_ind = torch.zeros(
                shape[0] * shape[2] * shape[3] 
            ).to(device=current_device, dtype=torch.long)
            batch_ind = torch.arange(
                0, 
                shape[0]
            ).repeat_interleave(shape[2] * shape[3]).long().to(device=current_device, dtype=torch.long)

            # Deform
            values = [
                map_tensor_coordinates(values[iV], indices, chan_ind, batch_ind).reshape(shape) 
                for iV in range(len(values))
            ]

        # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_deform_image_snapshot_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_deform_annot_snapshot_image.png'
        # )

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
                    values[iV][iC].astype('float32') / 255
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        else:
            values = [
               values[iV].astype('float32') / 255 
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
            if len(values[0].shape) == 2:
                values = [
                    np.where(values[iV] > self.cutoff, 1, 0).astype('uint8')
                    if iV in self.items
                    else values[iV]
                    for iV in range(len(values))
                ]
            elif len(values[0].shape) == 4:
                
                # Get current GPU device
                current_device = f'cuda:{values[0].get_device()}'
                one = torch.ones(1).to(device=current_device, dtype=torch.uint8)
                zero = torch.zeros(1).to(device=current_device, dtype=torch.uint8)
                values = [
                    torch.where(
                        values[iV] > self.cutoff, 
                        one, 
                        zero
                    )
                    if iV in self.items
                    else values[iV]
                    for iV in range(len(values))
                ]

        # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_binary_image_snapshot_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_binary_annot_snapshot_image.png'
        # )

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

        assert isinstance(std, float) and std >= 0 \
            or isinstance(std, int) and std == 0
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
        for iV in range(len(values)):
            if iV in self.items:
                if len(values[iV].shape) == 2:
                    values[iV] = np.clip(
                        values[iV] + np.random.normal(
                            scale=self.std, 
                            size=values[0].shape
                        ), 0, 1
                    ) 
                elif len(values[iV].shape) == 4:
                    current_device = f'cuda:{values[iV].get_device()}'
                    noise = torch.normal(
                        mean=0,
                        std=self.std, 
                        size=values[iV].shape
                    ).to(device=current_device, dtype=torch.float)
                    values[iV] = values[iV] + noise
                    values[iV] = torch.clip(values[iV], 0, 1) 

        # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_noise_image_snapshot_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_noise_annot_snapshot_image.png'
        # )

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
        elif len(values[0].shape) > 2:
            values = [
                np.expand_dims(values[iV], axis=1) 
                for iV in range(len(values))
            ]
            values = [
                torch.from_numpy(values[iV]) 
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


class ToNormal(object):
    """Crops a numpy array regularly to get a list of crops
    comprising the entire array
    """

    def __init__(self, items, new_mean, new_std):
        """ Args:
            input_size (int/tuple): input image sizes
            output_size (int/tuple): output image size
        """    

        assert isinstance(items, list)
        self.items = items

        assert isinstance(new_std, float)
        assert new_std > 0 and new_std < 1
        self.new_std = new_std
        
        assert isinstance(new_mean, float)
        assert new_mean > 0 and new_mean < 1
        self.new_mean = new_mean

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
        
        old_mean = torch.mean(values[0])
        old_std = torch.std(values[0])

        # Transform to new normal distribution
        values = [
            ((values[iV] - old_mean) / old_std) * self.new_std + self.new_mean
            if iV in self.items
            else values[iV]
            for iV in range(len(values)) 
        ] 

        # Clip
        values = [
            torch.clip(values[iV], 0, 1)
            for iV in range(len(values)) 
        ] 

        # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_normal_image_snapshot_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_normal_annot_snapshot_image.png'
        # )
         
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

        # Determine cropping positions
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
            if len(values[iV].shape) == 2:
                for top in tops:
                    for left in lefts:
                        cropped_samples[iV].append(values[iV][
                            top: top + self.output_size[0],
                            left: left + self.output_size[1]
                        ])
            elif len(values[iV].shape) == 4:
                for top in tops:
                    for left in lefts:
                        cropped_samples[iV].append(values[iV][
                            :,
                            :,
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
                if len(values[iV][iC].shape) == 4:
                    values[iV][iC] = torch.cat((
                        values[iV][iC],
                        torch.flip(values[iV][iC], [2]),
                        torch.flip(values[iV][iC], [3]),
                        torch.flip(values[iV][iC], [2, 3])
                    ), dim=1)

                    values[iV][iC] = torch.cat((
                        values[iV][iC],
                        torch.rot90(values[iV][iC], 1, [2, 3])
                    ), dim=1)

                elif len(values[iV][iC].shape) == 2:

                    mirrored_stack = np.stack((
                        values[iV][iC],
                        np.flip(values[iV][iC], 0),
                        np.flip(values[iV][iC], 1),
                        np.flip(values[iV][iC], (0, 1))
                    ), axis=0)
                    
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
                if len(values[iV][iC].shape) == 4:
                    values[iV][iC][:, 4:, :, :] = torch.rot90(
                        values[iV][iC][:, 4:, :, :], 
                        3, 
                        [2, 3]
                    )
                    values[iV][iC] = torch.cat((
                        values[iV][iC][:, 0, :, :],
                        torch.flip(values[iV][iC][:, 1, :, :], [2]),
                        torch.flip(values[iV][iC][:, 2, :, :], [3]),
                        torch.flip(values[iV][iC][:, 3, :, :], [2, 3]),
                        values[iV][iC][:, 4, :, :],
                        torch.flip(values[iV][iC][:, 5, :, :], [2]),
                        torch.flip(values[iV][iC][:, 6, :, :], [3]),
                        torch.flip(values[iV][iC][:, 7, :, :], [2, 3])
                    ), dim=1)
                else:
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
                    ), axis=0)

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
                torch.mean(values[iV][iC], dim=1)
                if len(values[iV][iC].shape) == 4
                else np.mean(values[iV][iC], axis=0)
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
            np.zeros(
                (
                    n_crops_h*n_crops_w, 
                    self.output_size[0], 
                    self.output_size[1]
                ), 
                dtype=float
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
        prediction = [
            np.where(
                prediction[iV] == 0,
                np.nan,
                prediction[iV]
            )
            for iV in range(len(values))
        ]

        # Merge
        prediction = [
            np.nanmean(prediction[iV], axis=0)
            for iV in range(len(values))
        ]

        return {
            list(sample.keys())[iV]: prediction[iV] 
            for iV in range(len(values))
        }