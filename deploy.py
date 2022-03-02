import torch
import torchvision.transforms as transforms
from FusionNet import * 
from datasets import LIVECell
from image_transforms import RandomCrop
from image_transforms import RandomOrientation
from image_transforms import BoundaryExtension
from image_transforms import Normalize
from image_transforms import ToTensor

def deploy ():  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\tUsing {device.upper()} device')

    # Disable grad
    with torch.no_grad():

        data_subset = 'val'
        data_folder = r'C:/Users/Max-S/tndrg/Data/LIVECell/'
        model_folder = r'C:/Users/Max-S/tndrg/Models/FusionNet_Pytorch/model/'
        load_snapshot = 10000

        #result_folder = r'C:/Users/Max-S/tndrg/Models/FusionNet_Pytorch/result/'

        LIVECell_test_dset = LIVECell(data_folder=data_folder,
                            data_subset=data_subset,
                            transform=transforms.Compose([
                                                    RandomCrop(output_size=512),
                                                    RandomOrientation(),
                                                    BoundaryExtension(ext=64),
                                                    Normalize(),
                                                    ToTensor()
                                                ]))
        
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