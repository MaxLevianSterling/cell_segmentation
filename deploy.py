import os
import torch
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.transforms as transforms
from torch.autograd import Variable
from FusionNet import * 
from datasets import LIVECell
from image_transforms import Normalize
from image_transforms import ToTensor
from utils import path_gen


def train(
    path = 'C:/Users/Max-S/tndrg',
    data_set = 'LIVECell',
    data_subset = 'trial',
    model = '1',
    load_snapshot = 50,
    save_snapshots = True,
    batch_size = 16,
    num_workers = 2, 
    pin_memory = True, 
    persistent_workers = True,
    lr = .0002,
    epochs = 50,
    verbosity_interval = 1,
    save_image_interval = 10,
    save_snapshot_interval = 1000,
):  
    """Trains a FusionNet-type neural network
    
    Note:
        Required folder structure:
            <path>/
                data/
                    <data_set>/
                        images/
                            <data_subset>/<.tif files>
                        annotations/
                            <data_subset>/<.tif files>
                            (<.json file>)
                models/
                results/
        Neural network uses grayscale input

    Args:
        path (string): training data folder
        data_set (string): training data set 
        data_subset (string): training data subset 
        model (string): current model identifier
        load_snapshot (int): training epoch age of saved 
            snapshot to start training at (default = 0 
            -> no snapshot loading)
        save_snapshots (bool): whether to save network
            snapshots at the specified interval (default = True)
        batch_size (int): data batch size (default = 16)
        num_workers (int): enables multi-process data loading 
            with the specified number of loader worker 
            processes (default = 2) 
        pin_memory (bool): tensors fetched by DataLoader pinned 
            in memory, enabling faster data transfer to 
            CUDA-enabled GPUs.(default = True)
        persistent_workers (bool): worker processes will not be 
            shut down after a dataset has been consumed once, 
            allowing the workers Dataset instances to remain 
            alive. (default = True)
        lr (float): network learning rate (default = .0002)
        epochs (int): number of training epochs (default = 50)
        verbosity_interval (int): epoch interval with which loss
            is displayed (default = 1)
        save_image_interval (int): epoch interval with which 
            example output is saved (default = 10)
        save_snapshot_interval (int): epoch interval with which 
            network snapshot is saved (default = 1000)
        
    Returns:
        (Optional) Example network image outputs at various training 
            stages in the training results subfolder
        (Optional) FusionNet snapshots in the models subfolder along 
            with a log of the loss over epochs
    """
    
    # Generate folder path strings
    models_folder = path_gen([
        path,
        'models',
        data_set,
        data_subset,
        model
    ])
    results_folder = path_gen([
        path,
        'results',
        data_set,
        data_subset,
        model,
        'training'
    ])

    # Create output directories if missing
    if not os.path.isdir(models_folder):
        os.makedirs(models_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    
    # Determine whether to use CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\tUsing {device.upper()} device')

    # Disable gradient calculation
    with torch.no_grad():

        # Initiate custom DataSet() instance
        LIVECell_deploy_dset = LIVECell(
            path=path,
            data_set=data_set,
            data_subset=data_subset,
            deploy=True,
            transform=transforms.Compose([
                Normalize(),
                ToTensor()
            ])
        )
        
        # Retrieve item
        index = 33
        sample = LIVECell_test_dset[index]
        image = sample['image']
        annot = sample['annot']

        # Loading the saved model
        model_path = f'{model_folder}FusionNet_snapshot{load_snapshot}.pkl'
        FusionNet = nn.DataParallel(FusionGenerator(1,1,64)).to(device) 
        FusionNet.load_state_dict(torch.save(model_path))
        FusionNet.eval()

        # Generate prediction
        for image in images:
            for crop in crops:
                for orientation in orientations:
                    image = 
                    prediction = FusionNet(image)
                    prediction = prediction[64:512+64-1,
                                            64:512+64-1]

        # Predicted class value using argmax
        predicted_class = np.argmax(prediction)

        # Reshape image
        image = image.reshape(28, 28, 1)

        # Show result
        plt.imshow(image, cmap='gray')
        plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
        plt.show()

if __name__ == '__main__':
    deploy()