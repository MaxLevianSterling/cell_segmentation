import torch
import numpy            as np
from utils              import path_gen
from utils              import reset_seeds


class LIVECell():
    """Custom dataset class to handle LIVECell data
        
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
        epoch_posttransforms=None,
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
        self.epoch_posttransforms = epoch_posttransforms

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
        dataset = {}
        dataset['image'] = np.load(f'{image_folder}array.npy')
        if not self.deploy:
            dataset['annot'] = np.load(f'{annot_folder}array.npy')

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
            with torch.no_grad():
                self.dataset = offline_transforms(dataset)

        # Send to GPU
        if not self.deploy:
            self.dataset = {
                key:self.dataset[key].to(device=dataset_device)
                for key in self.dataset.keys()
            }

    def epoch_pretransform(self):
        
        # Reset all randomness
        reset_seeds()

        # Perform online epoch pretransforms
        with torch.no_grad():
            self.epoch_dataset = self.epoch_pretransforms(self.dataset)
    
    def epoch_posttransform(self, predictions):
        
        # Reset all randomness
        reset_seeds()

        # Perform online epoch pretransforms
        with torch.no_grad():
            self.predictions = self.epoch_posttransforms(predictions)

        return self.predictions

    def __len__(self):
        """Define Dataset() length"""

        return len(self.image_filenames)
        
    def __call__(self, batch_idxs):
        """Upon subscription or iteration
        
        Args:
            batch_idxs (tensor): sample/annotation indices
        """

        # If annotation data is relevant
        if self.deploy:

            # Make image batch
            batch = {
                'image': [
                    torch.index_select(
                        self.epoch_dataset['image'][iC], 
                        0, 
                        batch_idxs
                    )
                    for iC in range(len(self.epoch_dataset['image']))
                ]
            }
        
        else:

            # Make image/annotation batch
            batch = {
                'image': torch.index_select(
                    self.epoch_dataset['image'], 
                    0, 
                    batch_idxs
                ),
                'annot': torch.index_select(
                    self.epoch_dataset['annot'], 
                    0,
                    batch_idxs
                ),
            }

        return batch