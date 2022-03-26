import torch
import numpy            as np
from torch.utils.data   import Dataset
from utils              import path_gen


class LIVECell(Dataset):
    """Custom PyTorch Dataset() class to handle LIVECell data
        
    Note:
        Image data must be grayscale
        Required folder structure:
            <path>/
                data/
                    <data_set>/
                        images/
                            variables/
                        annotations/
                            variables/
    """
    
    def __init__(
        self, 
        path, 
        data_set, 
        data_subset, 
        dataset_device,
        offline_transforms=None,
        epoch_pretransforms=None,
        deploy=False, 
    ):
        """Args:
            path (string): path to training data folder
            data_set (string): training data set
            data_subset (string): training data subset
            deploy (bool): whether the dataset will be used for
                deployment
        """

        # Initialize class variables
        self.deploy = deploy
        self.dataset_device = dataset_device
        self.epoch_pretransforms = epoch_pretransforms

        # Generate folder path strings
        image_folder = path_gen([
            path, 
            'data', 
            data_set, 
            'images', 
            data_subset, 
            'variables'
        ])
        if not self.deploy:
            annot_folder = path_gen([
                path, 
                'data', 
                data_set, 
                'annotations', 
                data_subset, 
                'variables'
            ])

        # Load data
        data = {}
        data['image'] = np.load(f'{image_folder}array.npy')
        if not self.deploy:
            data['annot'] = np.load(f'{annot_folder}array.npy')

        # Initialize internal variables
        self.image_filenames = []
        if not self.deploy:
            self.annot_filenames = []
       
        # Load filename identifiers
        with open(f'{image_folder}filenames.txt', 'r') as infile:
            for line in infile:
                args = line.split('\n')
                self.image_filenames.append(args[0])
        if not self.deploy:
            with open(f'{annot_folder}filenames.txt', 'r') as infile:
                        for line in infile:
                            args = line.split('\n')
                            self.annot_filenames.append(args[0])

        # Perform offline transform
        if offline_transforms:
            self.dataset = offline_transforms(data)

        # Send to GPU
        self.dataset = {
            key:self.dataset[key].to(device=dataset_device)
            for key in self.dataset.keys()
        }

    def __len__(self):
        """Define Dataset() length"""

        return len(self.image_filenames)

    def epoch_pretransform(self):
        
        # Perform online epoch pretransforms
        self.epoch_dataset = self.epoch_pretransforms(self.dataset)
        
        # Send to CPU
        self.epoch_dataset = {
            key:self.epoch_dataset[key].to(device='cpu')
            for key in self.dataset.keys()
        }

    def __getitem__(self, idx):
        """Upon subscription or iteration
        
        Args:
            idx (int/tensor): sample/annotation index
        """

        # If annotation data is relevant
        if not self.deploy:

            # Match image and annotation identifier
            identifier = self.image_filenames[idx]
            annot_idx = self.annot_filenames.index(identifier)

            # Make image/annotation sample
            sample = {
                'image': self.epoch_dataset['image'][idx, 0:1, :, :], 
                'annot': self.epoch_dataset['annot'][annot_idx, 0:1, :, :]
            }
                      
        else:

            # Make image sample
            sample = 1

        return sample