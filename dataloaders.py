from torch      import randperm, arange
from math       import ceil, floor


class GPU_dataloader():
    """ Data loader that provides batches by combining 
        a GPU dataset and a sampler 
    """

    def __init__(
        self, 
        dataset, 
        dataset_device,
        batch_size  = 1,
        shuffle     = False, 
        drop_last   = False,
    ):
        """ Args:
            dataset (Dataset): dataset from which to load
            batch_size (int): how many samples per batch 
                to load
            shuffle (bool): whether to reshuffle data every 
                epoch
            drop_last (bool): whether to drop the last 
                incomplete batch if the dataset size is not 
                divisible by the batch size. 
        """

        self.dataset = dataset
        self.dataset_device = dataset_device
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Shuffle data if necessary
        if shuffle:
            self.idxs = randperm(
                len(self.dataset)
            ).to(device=self.dataset_device)
        else:
            self.idxs = arange(
                len(self.dataset)
            ).to(device=self.dataset_device)

    def __len__(self):

        # Determine loader length
        loader_len = len(self.dataset) / self.batch_size
        if self.drop_last:
            loader_len = floor(loader_len)
        else:
            loader_len = ceil(loader_len)
        
        return loader_len

    def __iter__(self):

        # Copy epoch indices
        self.epoch_idxs = self.idxs

        return self

    def __next__(self):
        
        # Get batch indices
        batch_idxs = self.epoch_idxs[:self.batch_size]

        # Determine batch length
        current_batch_len = list(batch_idxs.size())[0]

        # Stop iteration when empty or dropped as last
        if current_batch_len < self.batch_size:
            if self.drop_last or current_batch_len == 0:
                raise StopIteration
        
        # Remove used indices
        self.epoch_idxs = self.epoch_idxs[self.batch_size:]

        # Get batch data
        batch = self.dataset(batch_idxs)

        return batch
