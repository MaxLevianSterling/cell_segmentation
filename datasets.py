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
        deploy=False, 
        transform=None
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
        self.transform = transform

        # Generate image folder path string
        image_folder = path_gen([
            path, 
            'data', 
            data_set, 
            'images', 
            data_subset, 
            'variables'
        ])

        # Load image data
        self.image_arr = np.load(f'{image_folder}array.npy')

        # Initialize internal variable
        self.image_filenames = []

        # Load image filename identifiers
        with open(f'{image_folder}filenames.txt', 'r') as infile:
            for line in infile:
                args = line.split('\n')
                self.image_filenames.append(args[0])

        # If annotation data is relevant
        if not self.deploy:
            
            # Generate annotation folder path string
            annot_folder = path_gen([
                path, 
                'data', 
                data_set, 
                'annotations', 
                data_subset, 
                'variables'
            ])

            # Load annotation data
            self.annot_arr = np.load(f'{annot_folder}array.npy')
            
            # Initialize internal variable
            self.annot_filenames = []
            
            # Load annotation filename identifiers
            with open(f'{annot_folder}filenames.txt', 'r') as infile:
                for line in infile:
                    args = line.split('\n')
                    self.annot_filenames.append(args[0])

    def __len__(self):
        """Define Dataset() length"""

        return len(self.image_filenames)
   
    def __getitem__(self, idx):
        """Upon subscription or iteration
        
        Args:
            idx (int/tensor): sample/annotation index
        """

        # Convert identifier tensor to list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # If annotation data is relevant
        if not self.deploy:

            # Match image and annotation identifier
            identifier = self.image_filenames[idx]
            annot_idx = self.annot_filenames.index(identifier)

            # Make image/annotation sample
            sample = {'image': self.image_arr[idx, :, :], 
                      'annot': self.annot_arr[annot_idx, :, :]}
                      
        else:

            # Make image sample
            sample = {'image': self.image_arr[idx, :, :]}

        # Transform sample
        if self.transform:
            sample = self.transform(sample)

        return sample