import numpy as np
import torch
from utils import unpad

# # Overfit one batch
    # LIVECell.epoch_pretransform(LIVECell_train_dset)
    # batch = next(iter(train_dataloader))

    # # Display progress
    # epoch_ratio = (iE) / (n_epochs - 1)
    # sys.stdout.write('\r')
    # sys.stdout.write(
    #     "\tEpochs: [{:<{}}] {:.0f}%; Loss: {:.5f}    ".format(
    #         "=" * int(20*epoch_ratio), 20, 100*epoch_ratio,
    #         loss.item()
    #     )
    # )
    # sys.stdout.flush()

# # Get image coordinates when image is embedded in black
# true_points = np.argwhere(values[0])
# bottom_right = true_points.max(axis=0)

# # Original encoding act_fn
# act_fn = nn.LeakyReLU(0.2, inplace=True)

# # Check if annots match images
# for iI in range(batch['image'].shape[0]):
#     v_utils.save_image(
#         x[iI].detach().to('cpu').type(torch.float32),
#         f'{results_folder}FusionNet_image{iI}.png'
#     )
#     v_utils.save_image(
#         y_[iI].detach().to('cpu').type(torch.float32),
#         f'{results_folder}FusionNet_annot{iI}.png'
#     )

# # Time processes
# from timeit             import default_timer        as timer  
# start = timer()
# print(timer() - start) 

# v_utils.save_image(
#     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
#     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_image_snapshot_image.png'
# )
# v_utils.save_image(
#     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
#     f'/mnt/sdg/maxs/results/LIVECell/FusionNet_annot_snapshot_image.png'
# )

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


class ToFullInterval():
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


class Unpadding():
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


class RandomTest():
    def __call__(self, sample):
        
        # Get input dictionary values
        values = [
            value 
            for value in sample.values()
        ]   
        
        print(torch.randint(
            1000,
            (1, 5)
        ))

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


class ToUnitInterval():
    """ Converts a numpy array or list of numpy arrays 
    from uint8 range to the unit interval
    """

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
        if isinstance(values[0], list) \
                and isinstance(values[0][0], np.ndarray):
            values = [
                [
                    values[iV][iC].astype('float32') / 255
                    for iC in range(len(values[0]))
                ]
                for iV in range(len(values))
            ]
        elif isinstance(values[0], np.ndarray):
            values = [
               values[iV].astype('float32') / 255 
               for iV in range(len(values))
            ]

        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }

                
class ToEdge():
    """Normalizes input according to new parameters
    for the entire set"""

    def __init__(self, items, old_n_edge, new_n_edge, old_p_edge, new_p_edge, old_p, new_p):
        """ Args:
            items (list): list of sample value numbers 
                to convert to binary
            new_mean (float): the new set mean
            new_std (float): the new set standard deviation
            old_mean (float): the old set mean
            old_std (float): the old set standard deviation
        """    

        assert isinstance(items, list)
        self.items = items

        self.old_n_edge = old_n_edge
        self.new_n_edge = new_n_edge
        self.old_p_edge = old_p_edge
        self.new_p_edge = new_p_edge
        self.old_p = old_p
        self.new_p = new_p
        
    def __call__(self, sample):
        """ Args:
            sample (string:(HxW np.ndarray or BxCxHxW tensor) 
                dict): input images
                    
        Returns:
            (string:(HxW np.ndarray or BxCxHxW tensor) dict): 
                output images
        """   

        # Get input dictionary values        
        values = [
            value 
            for value in sample.values()
        ]
        
        if torch.is_tensor(values[0]) \
                and len(values[0].shape) == 4:
            if values[0].get_device() >= 0:
                current_device = f'cuda:{values[0].get_device()}'
            else:
                current_device = 'cpu'
                
            old_n_edge = torch.as_tensor(self.old_n_edge, device=current_device, dtype=torch.float)
            new_n_edge = torch.as_tensor(self.new_n_edge, device=current_device, dtype=torch.float)
            old_p_edge = torch.as_tensor(self.old_p_edge, device=current_device, dtype=torch.float)
            new_p_edge = torch.as_tensor(self.new_p_edge, device=current_device, dtype=torch.float)
            old_p = torch.as_tensor(self.old_p, device=current_device, dtype=torch.float)
            new_p = torch.as_tensor(self.new_p, device=current_device, dtype=torch.float)

            values = [
                torch.where(
                    values[iV] == old_p, 
                    new_p,
                    values[iV]
                )
                if iV in self.items
                else values[iV]
                for iV in range(len(values))
            ]
            values = [
                torch.where(
                    values[iV] == old_n_edge, 
                    new_n_edge,
                    values[iV]
                )
                if iV in self.items
                else values[iV]
                for iV in range(len(values))
            ]
            values = [
                torch.where(
                    values[iV] == old_p_edge, 
                    new_p_edge,
                    values[iV]
                )
                if iV in self.items
                else values[iV]
                for iV in range(len(values))
            ]

        #print(values[1][0,0:1,0:20,0:20])

        # # Plotting utility
        # v_utils.save_image(
        #     values[0][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/edge_image.png'
        # )
        # v_utils.save_image(
        #     values[1][0,0:1,:,:].detach().to('cpu').type(torch.float32), 
        #     f'/mnt/sdg/maxs/results/LIVECell/edge_annot.png'
        # )
         
        return {
            list(sample.keys())[iV]: values[iV] 
            for iV in range(len(values))
        }


def unpad(arr, pad_widths):
    """Unpads a numpy array
    
    Args:
        arr (np.array): input array
        pad_widths (int): pad widths to remove
 
    Returns:
        (np.array) unpadded array
    """

    # Initialize array slices
    slices = []

    # For every array dimension
    for c in pad_widths:

        # Take inverse of the far point
        e = None if c[1] == 0 else -c[1]

        # Create slice object for that dimension
        slices.append(slice(c[0], e))
    
    # Unpad
    return arr[tuple(slices)]



def inception_one(in_chan, out_chan):
    block = nn.Sequential(
        nn.Conv2d(
            in_chan, 
            int(out_chan/4), 
            kernel_size=1, 
            stride=1, 
            padding=0,
        ),
    )

    return block


def inception_three(in_chan, out_chan):
    block = nn.Sequential(
        # nn.Conv2d(
        #     in_chan, 
        #     int(in_chan/4), 
        #     kernel_size=1, 
        #     stride=1, 
        #     padding=0,
        # ),
        nn.Conv2d(
            int(in_chan), 
            int(out_chan/2), 
            kernel_size=3, 
            stride=1, 
            padding=1,
            padding_mode='reflect',
        ),
    )
    
    return block


def inception_five(in_chan, out_chan):
    block = nn.Sequential(
        # nn.Conv2d(
        #     in_chan, 
        #     int(in_chan/4), 
        #     kernel_size=1, 
        #     stride=1, 
        #     padding=0,
        # ),
        nn.Conv2d(
            int(in_chan), 
            int(out_chan/4), 
            kernel_size=5, 
            stride=1, 
            padding=2,
            padding_mode='reflect',
        ),
    )

    return block


def act_batch(act_fn, out_chan):
    """ Creates the basic FusionNet convolution 
        block
    
    Args:
        act_fn (nn.Module): activation function
        in_chan (int): input channel depth
        out_chan (int): output channel depth
 
    Returns:
        (nn.Sequential()) Basic convolution block
    """

    block = nn.Sequential(
        act_fn,
        nn.BatchNorm2d(out_chan),
    )

    return block