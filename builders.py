import torch
import numpy            as np
from utils              import path_gen
from utils              import reset_seeds


class Builder():
    """Custom dataset builder"""
    
    def __init__(
        self, 
        path, 
        data_set, 
        data_type,
        data_subset,
        subset_type,
        annot_type,
        dataset_device,
        supervised=True,
        supervised_offline_transforms=None,
        unsupervised_offline_transforms=None,
        epoch_pretransforms=None,
        epoch_posttransforms=None,
        deploy=False, 
    ):
        """Args:
            supervised (bool): whether training should be 
                supervised with the specified annotation style
                or self-supervised as an autoencoder
            path (string): path to data folder
            data_set (string): data set ( i.e. LIVECell)
            data_type (string): data type (i.e. per_celltype, 
                part_set, full_set)
            data_subset (string): data subset (e.g. BV2, 50%)
            subset_type (string): data subset type (i.e. train, 
                test, val)
            annot_type (string): annotation style (soma, thin
                membrane, thick membrane)
            dataset_device (string): what device to handle 
                data on
            supervised_offline_transforms (Compose(Transforms)):
                composed sequence of one-time image transforms 
                to be performed on the CPU if supervised mode
                is on
            unsupervised_offline_transforms (Compose(Transforms)):
                composed sequence of one-time image transforms 
                to be performed on the CPU if unsupervised mode
                is on
            epoch_pretransforms (Compose(Transforms)):
                composed sequence of online image transforms 
                to be performed on the GPU before every epoch
            epoch_posttransforms (Compose(Transforms)):
                composed sequence of online image transforms 
                to be performed on the GPU after every epoch
            deploy (bool): whether the dataset will be used for
                deployment
        """

        # Initialize class variables
        self.supervised = supervised
        self.epoch_pretransforms = epoch_pretransforms
        self.epoch_posttransforms = epoch_posttransforms
        self.deploy = deploy

        # Generate folder path strings
        image_folder = path_gen([
            path, 
            'data',
            data_set,
            data_type,
            data_subset,
            'images',
            subset_type,
            'variables'
        ])
        if self.supervised and not self.deploy:
            annot_folder = path_gen([
                path, 
                'data',
                data_set,
                data_type,
                data_subset,
                'annotations',
                subset_type,
                annot_type,
                'variables'
            ])

        # Load data
        dataset = {}
        dataset['image'] = np.load(f'{image_folder}array.npy')
        if self.supervised and not self.deploy:
            dataset['comp'] = np.load(f'{annot_folder}array.npy')
        if not self.supervised:
            dataset['comp'] = np.load(f'{image_folder}array.npy')

        # Perform offline transform
        with torch.no_grad():
            if supervised and supervised_offline_transforms:
                self.dataset = supervised_offline_transforms(dataset)
            elif not supervised and unsupervised_offline_transforms:
                self.dataset = unsupervised_offline_transforms(dataset)

        # Send to GPU
        if not self.deploy:
            self.dataset = {
                key:self.dataset[key].to(device=dataset_device)
                for key in self.dataset.keys()
            }

    def epoch_pretransform(self):
        """Define epoch-based image pretransformations"""

        # Reset all randomness
        reset_seeds()

        # Perform online epoch pretransforms
        with torch.no_grad():
            self.epoch_dataset = self.epoch_pretransforms(self.dataset)
           
    def epoch_posttransform(self, predictions):
        """Define epoch-based image posttransformations"""
        
        # Reset all randomness
        reset_seeds()

        # Perform online epoch pretransforms
        with torch.no_grad():
            self.predictions = self.epoch_posttransforms(predictions)

        return self.predictions

    def __len__(self):
        """Define dataset length"""

        return self.dataset['image'].shape[0]
        
    def __call__(self, batch_idxs):
        """When dataset class is called
        
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
                'comp': torch.index_select(
                    self.epoch_dataset['comp'], 
                    0,
                    batch_idxs
                ),
            }

        return batch