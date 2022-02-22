from FusionNet import * 
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from dataset import *

if __name__ == '__main__':  
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

    LIVECell = LIVECell(data_folder=data_folder,
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
    dataloader = data.DataLoader(LIVECell, 
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
