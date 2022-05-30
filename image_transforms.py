import torch
import warnings
import numpy                        as np
import torchvision.utils            as v_utils
import torch.nn.functional          as tfunc
from math                           import ceil
from utils                          import map_tensor_coordinates


class Compose():
    """ Composes several transforms together
   
    Warning:
        PyTorch random seed is automatically reset
            when a worker is initialized, but other
            random libraries not. Use custom 
            worker_init_fn or other function to 
            achieve this per epoch or otherwise.

    Examples:
    -   offline_transforms=Compose([
            ToTensor(type=torch.float),
            ToUnitInterval(items=[0, 1]),
            ToNormal(items=[0], new_mean=new_mean, new_std=new_std),
        ])
    -   epoch_pretransforms=Compose([
            RandomCrop(input_size=orig_size, output_size=crop_size),
            RandomOrientation(),
            LocalDeform(size=localdeform[0], ampl=localdeform[1]),
            ToBinary(cutoff=tobinary, items=[1]),
            Noise(std=noise, items=[0]),
        ])
    """

    def __init__(self, transforms):
        """ Args:
            transforms (list of objects): image transforms
        """      

        self.transforms = transforms

    def __call__(self, sample):
        """ Args:
            sample (string:BxCxHxW tensor dict): input images
                    
        Returns:
            (string:BxCxHxW tensor dict): output images
        """    

        # Conduct transforms sequentially
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def __repr__(self):
        """Generate custom string representation"""

        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class ToTensor():
    """ Converts a numpy array or list of numpy arrays to
    PyTorch tensor format
    """

    def __init__(self, type):
        """ Args:
            type (PyTorch datatype): output tensor datatype
        """

        self.type = type

    def __call__(self, sample):
        """ Args:
            sample (string:[np.ndarray] dict): input images
                    
        Returns:
            (string:tensor dict): output tensors
        """   

        # Get input dictionary values        
        values = [
            value 
            for value in sample.values()
        ]

        # Convert to tensor and expand channel dimension
        if isinstance(values[0], list) \
                and isinstance(values[0][0], np.ndarray):
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
        elif isinstance(values[0], np.ndarray) \
                and len(values[0].shape) == 2:
            values = [
                np.expand_dims(values[iV], axis=0) 
                for iV in range(len(values))
            ]
            values = [
                torch.from_numpy(values[iV]) 
                for iV in range(len(values))
            ]
        elif isinstance(values[0], np.ndarray) \
                and len(values[0].shape) == 3:
            values = [
                np.expand_dims(values[iV], axis=1) 
                for iV in range(len(values))
            ]
            values = [
                torch.from_numpy(values[iV]).type(self.type)
                for iV in range(len(values))
            ]
        else:
            raise RuntimeError('Wrong input format used')
        
        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/tensor_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/tensor_annot.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToUnitInterval():
    """ Converts a numpy array or list of numpy arrays 
    from uint8 range to the unit interval
    """

    def __init__(self, items):
        """ Args:
            items (list): list of sample value numbers 
                to convert to binary
        """    

        assert isinstance(items, list)
        self.items = items

    def __call__(self, sample):
        """ Args:
            sample (string:[np.ndarray] dict): input images
                    
        Returns:
            (string:[np.ndarray.float] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Transform to unit interval
        if torch.is_tensor(values[0]) \
                and len(values[0].shape) == 4:
            values = [
               values[iV] / 255 
               if iV in self.items
               else values[iV]
               for iV in range(len(values))
            ]
        else:
            raise RuntimeError('Wrong input type')

        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/unit_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/unit_annot.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToNormal():
    """Normalizes input according to new parameters
    for the entire set"""

    def __init__(self, items, new_mean, new_std):
        """ Args:
            items (list): list of sample value numbers 
                to convert to binary
            new_mean (float): the new set mean
            new_std (float): the new set standard deviation
        """    

        assert isinstance(items, list)
        self.items = items

        assert isinstance(new_mean, (int, float))
        self.new_mean = new_mean

        assert isinstance(new_std, (int, float))
        self.new_std = new_std
        
    def __call__(self, sample):
        """ Args:
            sample (string:BxCxHxW tensor dict): input images
                    
        Returns:
            (string:BxCxHxW tensor dict): output images
        """   

        # Get input dictionary values        
        values = [
            value 
            for value in sample.values()
        ]
        
        for iV in range(len(values)):
            if iV in self.items:

                # Get old distribution
                old_mean = torch.mean(values[iV])
                old_std = torch.std(values[iV])

                # Transform
                values[iV] = ((values[iV] - old_mean) / old_std) \
                    * self.new_std + self.new_mean

        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/normal_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/normal_annot.png'
        # )
         
        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }
       

class RandomCrop():
    """Randomly crops input"""

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
            sample (string:BxCxHxW tensor dict): input images
                    
        Returns:
            (string:BxCxHxW tensor dict): output images
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
            if torch.is_tensor(values[iV]) \
                    and len(values[iV].shape) == 4:
                values[iV] = values[iV][
                    :,
                    :,
                    top: top + self.output_size[0],
                    left: left + self.output_size[1]
                ]
            else:
                raise RuntimeError('Wrong input type')
        
        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/crop_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/crop_annot.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class RandomOrientation():
    """Randomly orients input into one of eight orientations 
        (all 90 degree rotations and mirrors)
    """

    def __call__(self, sample):
        """ Args:
            sample (string:BxCxHxW tensor dict): input images
                    
        Returns:
            (string:BxCxHxW tensor dict): output images
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
            if torch.is_tensor(values[iV]) \
                    and len(values[iV].shape) == 4:
                
                # Vertical flip
                if mirror > 2:
                    values[iV] = torch.flip(values[iV], [2])

                # Horizontal flip
                if mirror % 2 == 0:
                    values[iV] = torch.flip(values[iV], [3])

                # Counterclockwise rotation
                values[iV] = torch.rot90(
                    values[iV],
                    n_rotations.item(), 
                    [2, 3]
                )
            else:
                raise RuntimeError('Wrong input type')

        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/orient_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/orient_annot.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class LocalDeform():
    """Locally deforms input based on a randomly 
    generated sparse vector field
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
            sample (string:BxCxHxW tensor dict): input images
                    
        Returns:
            (string:BxCxHxW tensor dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Get input shape
        shape = values[0].shape

        if torch.is_tensor(values[0]) and len(shape) == 4:
            
            # Get current GPU device
            current_device = f'cuda:{values[0].get_device()}'
            
            # Initialize random sparse vector field
            dS = [
                torch.FloatTensor(
                    shape[0], 
                    1, 
                    self.size[0] - 2, 
                    self.size[1] - 2
                ).uniform_(
                    -self.ampl, 
                    self.ampl
                ).to(device=current_device, dtype=torch.float)
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
                        scale_factor=(
                            shape[2]/self.size[0], 
                            shape[3]/self.size[1]
                        ), 
                        align_corners=True, 
                        mode='bilinear'
                    )
                    for iS in range(2)
                ]       
        
            # Determine axis-wise pixel transformations
            X, Y = torch.meshgrid(
                torch.arange(shape[2]).to(
                    device=current_device, 
                    dtype=torch.float
                ), 
                torch.arange(shape[3]).to(
                    device=current_device, 
                    dtype=torch.float
                ), 
                indexing='xy'
            )
            X = X.unsqueeze(0).repeat(shape[0], 1, 1).unsqueeze(1)
            Y = Y.unsqueeze(0).repeat(shape[0], 1, 1).unsqueeze(1)
            indices = torch.reshape(Y+dS[1], (-1, 1)), \
                torch.reshape(X+dS[0], (-1, 1))
            indices = torch.cat((indices[0], indices[1]), dim=1)
            indices = torch.transpose(indices, 0, 1)

            # Create batch and channel indices
            chan_ind = torch.zeros(
                shape[0] * shape[2] * shape[3] 
            ).to(device=current_device, dtype=torch.long)
            batch_ind = torch.arange(
                0, 
                shape[0]
            ).repeat_interleave(shape[2] * shape[3]).long().to(
                device=current_device, 
                dtype=torch.long
            )

            # Deform
            values = [
                map_tensor_coordinates(
                    values[iV], 
                    indices, 
                    chan_ind, 
                    batch_ind
                ).reshape(shape) 
                for iV in range(len(values))
            ]
        else:
            raise RuntimeError('Wrong input type')

        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/deform_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/deform_annot.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToBinary():
    """Converts input from unit interval range to binary 
        based on a cutoff
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
            sample (string:BxCxHxW tensor dict): input images
                    
        Returns:
            (string:BxCxHxW tensor dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]
        #valuess = [copy.deepcopy(values) for i in range(19)]
        # Transform to binary as per cutoff
        #for i in range(19):
        if torch.is_tensor(values[0]) \
                and len(values[0].shape) == 4:
            if values[0].get_device() >= 0:
                current_device = f'cuda:{values[0].get_device()}'
            else:
                current_device = 'cpu'
            one = torch.ones(1).to(
                device=current_device, 
                dtype=torch.uint8
            )
            zero = torch.zeros(1).to(
                device=current_device, 
                dtype=torch.uint8
            )
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
        else:
            raise RuntimeError('Wrong input type')

        # Plotting utility
        # for i in range(1,20):
        #     v_utils.save_image(
        #         valuess[i-1][0][11,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #         f'/mnt/sdg/maxs/results/LIVECell/binary_image_0{i/20}_11.png'
        #     )
        #     v_utils.save_image(
        #         valuess[i-1][0][21,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #         f'/mnt/sdg/maxs/results/LIVECell/binary_image_0{i/20}_21.png'
        #     )
        #     v_utils.save_image(
        #         valuess[i-1][0][66,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #         f'/mnt/sdg/maxs/results/LIVECell/binary_image_0{i/20}_66.png'
        #     )
        #     v_utils.save_image(
        #         valuess[i-1][0][76,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #         f'/mnt/sdg/maxs/results/LIVECell/binary_image_0{i}_76.png'
        #     )
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/binary_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/binary_annot.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class Noise():
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
            sample (string:BxCxHxW tensor dict): input images
                    
        Returns:
            (string:BxCxHxW tensor dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Add noise and clip at unit interval
        for iV in range(len(values)):
            if iV in self.items:
                if torch.is_tensor(values[0]) \
                        and len(values[0].shape) == 4:
                    current_device = f'cuda:{values[iV].get_device()}'
                    noise = torch.normal(
                        mean=0,
                        std=self.std, 
                        size=values[iV].shape
                    ).to(device=current_device, dtype=torch.float)
                    values[iV] = values[iV] + noise
                else:
                    raise RuntimeError('Wrong input type')

        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/noise_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/noise_annot.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }
  
        
class FullCrop():
    """Crops a numpy array regularly to get a list of crops
    comprising the entire array
    """

    def __init__(self, input_size, output_size, overlap):
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
       
        assert isinstance(overlap, (int, tuple))
        if isinstance(overlap, int):
            self.overlap = (overlap, overlap)
        else:
            assert len(output_size) == 2
            self.overlap = overlap

    def __call__(self, sample):
        """ Args:
            sample (string:np.ndarray dict): input images
                    
        Returns:
            (string:[np.ndarrays] dict): output images
        """   

        # Get input dictionary values        
        values = [
            value 
            for value in sample.values()
        ]

        # Determine how many crops needed for each axis
        n_crops_h = ceil(
            self.overlap[0] * (self.input_size[0] - self.output_size[0]
        ) / self.output_size[0]) + 1
        n_crops_w = ceil(
            self.overlap[1] * (self.input_size[1] - self.output_size[1]
        ) / self.output_size[1]) + 1

        # Determine cropping positions
        tops = [
            round(iCh * (self.input_size[0]-self.output_size[0]) / (n_crops_h-1)) 
            for iCh in range(n_crops_h)
        ]
        lefts = [
            round(iCw * (self.input_size[1]-self.output_size[1]) / (n_crops_w-1)) 
            for iCw in range(n_crops_w)
        ]
       
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


class StackOrient():
    """Creates a stack of all 8 numpy array orientations   
    attainable by a combination of 90 degree rotations 
    in place of each numpy array input list item
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.ndarray] dict): input images
                    
        Returns:
            (string:[np.ndarrays] dict): output images
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

        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/oriented_image0.png'
        # )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class StackReorient():
    """Reorients a list of numpy array stack of all 8 
    orientations attainable by a combination of 90 degree 
    rotations back into their original position
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.ndarray] dict): input images
                    
        Returns:
            (string:[np.ndarray] dict): output images
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
                        values[iV][iC][:, 0:1, :, :],
                        torch.flip(values[iV][iC][:, 1:2, :, :], [2]),
                        torch.flip(values[iV][iC][:, 2:3, :, :], [3]),
                        torch.flip(values[iV][iC][:, 3:4, :, :], [2, 3]),
                        values[iV][iC][:, 4:5, :, :],
                        torch.flip(values[iV][iC][:, 5:6, :, :], [2]),
                        torch.flip(values[iV][iC][:, 6:7, :, :], [3]),
                        torch.flip(values[iV][iC][:, 7:8, :, :], [2, 3])
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
        
        # Plotting utility
        v_utils.save_image(
            values[0][5][10,0:1,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image0.png'
        )
        v_utils.save_image(
            values[0][5][10,1:2,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image1.png'
        )
        v_utils.save_image(
            values[0][5][10,2:3,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image2.png'
        )
        v_utils.save_image(
            values[0][5][10,3:4,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image3.png'
        )
        v_utils.save_image(
            values[0][5][10,4:5,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image4.png'
        )
        v_utils.save_image(
            values[0][5][10,5:6,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image5.png'
        )
        v_utils.save_image(
            values[0][5][10,6:7,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image6.png'
        )
        v_utils.save_image(
            values[0][5][10,7:8,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/reoriented_image7.png'
        )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class StackMean():
    """Averages a list of numpy array stacks across the 0th 
    dimension
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.ndarray] dict): input images
                    
        Returns:
            (string:[np.ndarray] dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]
        
        # Average
        values = [
            [   
                torch.mean(values[iV][iC], dim=1, keepdim=True)
                if len(values[iV][iC].shape) == 4
                else np.mean(values[iV][iC], axis=0)
                for iC in range(len(values[0]))
            ]
            for iV in range(len(values))
        ]

        # Plotting utility
        v_utils.save_image(
            values[0][5][10,0:1,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/mean_image0.png'
        )

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }
        
        
class Uncrop():
    """Merges a list of regularly cropped numpy arrays
    back into an averaged array in the shape of the
    original array
    """

    def __init__(self, input_size, output_size, overlap):
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
       
        assert isinstance(overlap, (int, tuple))
        if isinstance(overlap, int):
            self.overlap = (overlap, overlap)
        else:
            assert len(output_size) == 2
            self.overlap = overlap

    def __call__(self, sample):
        """ Args:
            sample (string:[np.ndarray] dict): input images
                    
        Returns:
            (string:np.ndarray dict): output images
        """   

        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]

        # Determine how many crops needed for each axis
        n_crops_h = ceil(
            self.overlap[0] * (self.output_size[0] - self.input_size[0]
        ) / self.input_size[0]) + 1
        n_crops_w = ceil(
            self.overlap[1] * (self.output_size[1] - self.input_size[1]
        ) / self.input_size[1]) + 1

        # Determine cropping positions
        tops = [
            round(iCh * (self.output_size[0]-self.input_size[0]) / (n_crops_h-1)) 
            for iCh in range(n_crops_h)
        ]
        lefts = [
            round(iCw * (self.output_size[1]-self.input_size[1]) / (n_crops_w-1)) 
            for iCw in range(n_crops_w)
        ]

        if isinstance(values[0], list) \
                and isinstance(values[0][0], np.ndarray) \
                and len(values[0][0].shape) == 3:

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

        elif isinstance(values[0], list) \
                and torch.is_tensor(values[0][0]) \
                and len(values[0][0].shape) == 4:
    
            # Initialize output array
            prediction = [
                -1 * torch.ones(
                    (
                        n_crops_h*n_crops_w,
                        values[iV][0].shape[0], 
                        1,
                        self.output_size[0], 
                        self.output_size[1],
                    ), 
                    dtype=torch.float
                )
                for iV in range(len(values))
            ]

            # Pad crops with zeros
            for iV in range(len(values)):
                for iT, top in enumerate(tops):
                    for iL, left in enumerate(lefts):
                        prediction[iV][
                            len(lefts)*iT+iL : len(lefts)*iT+iL+1,
                            :,
                            :,
                            top: top + self.input_size[0],
                            left: left + self.input_size[1],
                        ] = values[iV][len(lefts)*iT+iL] 
            
            # Transform zero to NaN
            nan = torch.zeros(1)
            nan[nan==0] = float('nan')
            prediction = [
                torch.where(
                    prediction[iV] == float(-1),
                    nan,
                    prediction[iV]
                )
                for iV in range(len(values))
            ]

            # Merge
            prediction = [
                torch.nanmean(prediction[iV], axis=0)
                for iV in range(len(values))
            ]

        # Plotting utility
        v_utils.save_image(
            prediction[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/mean_image0.png'
        )
        v_utils.save_image(
            prediction[0][1,0:1,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/mean_image1.png'
        )
        v_utils.save_image(
            prediction[0][2,0:1,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/mean_image2.png'
        )
        v_utils.save_image(
            prediction[0][3,0:1,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/mean_image3.png'
        )
        v_utils.save_image(
            prediction[0][4,0:1,:,:].detach().to('cpu').type(torch.float32), 
            f'/mnt/sdg/maxs/results/LIVECell/mean_image4s.png'
        )

        return {
            list(sample.keys())[iV]: prediction[iV] 
            for iV in range(len(values))
        }


class Squeeze():
    """Squeezes a numpy array or list of numpy arrays 
    in the 1st dimension
    """

    def __call__(self, sample):
        """ Args:
            sample (string:[np.ndarray] dict): input images
                    
        Returns:
            (string:[np.ndarray] dict): output images
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
