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