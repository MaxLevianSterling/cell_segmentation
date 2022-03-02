import torch
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.transforms as transforms
from torch.autograd import Variable
from FusionNet import * 
from datasets import LIVECell
from image_transforms import RandomCrop
from image_transforms import RandomOrientation
from image_transforms import LocalDeform
from image_transforms import BoundaryExtension
from image_transforms import Normalize
from image_transforms import Noise
from image_transforms import ToTensor


def train():  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\tUsing {device.upper()} device')

    batch_size = 16
    lr = 0.0002 
    epochs = 2
    verbosity_interval = 1
    save_image_interval = 1
    save_snapshot_interval = 1000
    load_snapshot = 0

    data_subset = 'trial'
    data_folder = r'C:/Users/Max-S/tndrg/Data/LIVECell/'
    model_folder = r'C:/Users/Max-S/tndrg/Models/FusionNet_Pytorch/model/'
    result_folder = r'C:/Users/Max-S/tndrg/Models/FusionNet_Pytorch/result/'

    LIVECell_train_dset = LIVECell(data_folder=data_folder,
                        data_subset=data_subset,
                        transform=transforms.Compose([
                                                RandomCrop(output_size=512),
                                                RandomOrientation(),
                                                LocalDeform(size=12,ampl=8), #Introduces nonbinary elements into annotation tensor
                                                BoundaryExtension(ext=64),
                                                Normalize(),
                                                Noise(std=.1),
                                                ToTensor()
                                            ]))
    dataloader = data.DataLoader(LIVECell_train_dset, 
                                batch_size=batch_size,
                                shuffle=True, 
                                num_workers=0, #num_workers=2
                                pin_memory=False, #True
                                persistent_workers=False) #True
                                # Custom batch_sampler + collate_fn?

    FusionNet = nn.DataParallel(FusionGenerator(1,1,64)).to(device) 
    FusionNet = FusionNet.float()
    # FusionNet = nn.DataParallel(FusionGenerator(1,1,64)).cuda() 
    # implement DISTRIBUTEDDATAPARALLEL() instead?

    try:
        FusionNet = torch.load(f'{model_folder}FusionNet_{load_snapshot}.pkl')
        print('\nSnapshot at epoch {load_snapshot} restored')
    except:
        pass

    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(FusionNet.parameters(),lr=lr)

    for epoch in range(epochs):
        for iter, batch in enumerate(dataloader):

            optimizer.zero_grad()

            x = Variable(batch['image']).to(device)
            y_ = Variable(batch['annot']).to(device)
            y = FusionNet.forward(x.float())
            
            loss = loss_func(y, y_)
            loss.backward()
            optimizer.step()
 
            if epoch % verbosity_interval == 0 and iter == 0:
                print(f'Epoch: {epoch+1}/{epochs}; Loss: {loss}')

            if epoch % save_image_interval == 0 and iter == 0:
                v_utils.save_image(x[0].cpu().data, f'{result_folder}original_image_snapshot{load_snapshot}_epoch{epoch}.png')
                v_utils.save_image(y_[0].cpu().data, f'{result_folder}label_image_snapshot{load_snapshot}_epoch{epoch}.png')
                v_utils.save_image(y[0].cpu().data, f'{result_folder}gen_image_snapshot{load_snapshot}_epoch{epoch}.png')
            
            if epoch % save_snapshot_interval == 0:
                torch.save(FusionNet, f'{model_folder}FusionNet_snapshot{load_snapshot}.pkl')    


if __name__ == '__main__':
    train()