import os
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
import seaborn              as sns
from utils                  import path_gen


def plotting(
    path = '/mnt/sdg/maxs',
    data_sets = ['LIVECell','LIVECell','LIVECell'],
    data_subsets = ['train','train','train'],
    print_separator = '$',
    models = ['1','2','4'],
    epoch_range = [0, 0],
    mode = 'loss'
):
    """Plots FusionNet data in different ways

    Note:
        Image data must be grayscale
        Required folder structure:
            <path>/
                results/
                    <data_set>/
                        <data_subset>/
                            <model>/
                plots/

    Args:
        path (string): path to training data folder
        data_sets (list of strings): data sets to be plotted
        data_subsets (list of strings): data subsets to be 
            plotted
        print_separator (string): print output separation
            character
        models (list of strings): models to be plotted
        epoch_range (list of ints): epoch range to be plotted
        mode (string): function mode can only be 'loss' (plots 
            model losses over time)
       
    Returns:
        Various plots in the plots subfolder
    """

    # Being beautiful is not a crime
    print('\n', f'{print_separator}' * 87, '\n', sep='')

    # Generate plots folder path string
    plots_folder = path_gen([
        path,
        'plots',
    ])

    if mode == 'loss':

        model_names = [
            f'{data_sets[iM]}_dataset_{data_subsets[iM]}_datasubset__model_{model}'
            for iM, model in enumerate(models)
        ]

        results_folders = [
            path_gen([
                path,
                'results',
                data_sets[iM],
                data_subsets[iM],
                model,
                'training'
            ])
            for iM, model in enumerate(models)
        ]

        all_loss_files = []
        for result_folder in results_folders:
            files = os.listdir(result_folder)
            loss_files = [
                file 
                for file in files
                if file.endswith('.txt')
            ]
            all_loss_files.append(loss_files)

        #TODO: Append different snapshot files together automatically
        #TODO: and use the new epoch indices to do that

        losses = [[] for model in models]
        for iM in range(len(models)):
            with open(f'{results_folders[iM]}{all_loss_files[iM][0]}', 'r') as infile:
                for line in infile:
                    line = line.strip('\n')
                    losses[iM].append(float(line))

        losses[2].extend([0.0 for i in range(300)])
        for i in range(25):
            losses[0].insert(0, 0.0)
        for i in range(150):
            losses[1].insert(0, 0.0)
        losses[1].extend([0.0 for i in range(25)])

        losses_df = pd.DataFrame(
            {
                model_name: loss 
                for model_name, loss in zip(model_names, losses)
            }
        )
        
        losses_df.sample(5)
        a=1
    
plotting()

# # Some modules in the code subfolder use 'Agg', 
# #       which is not compatible with plotting 
# #       grayscale images. This import snippet 
# #       ensures grayscale plotting is possible.
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt

# # Grayscale plotting
# plt.imshow(array, cmap='gray', vmin=0, vmax=255)
# plt.show()

# # Grayscale plotting with my image_transforms variables
# for iV in range(len(values)):
#         plt.imshow(values[iV], cmap='gray', vmin=0, vmax=255)
#         plt.show()

# # Plot the sparse vector field of the LocalDeform class
# plt.quiver(du, dv)
# plt.axis('off')
# plt.show()
# plt.clf()

# # LocalDeform plotting for image and annotation
# plt.imshow(np.hstack( (np.squeeze(image), 
#                         np.squeeze(deformed_image), 
#                         np.squeeze(image-deformed_image)
#                         ), 
#                         ), cmap = plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()

# plt.imshow(np.hstack( (np.squeeze(annot), 
#                         np.squeeze(deformed_annot), 
#                         np.squeeze(annot-deformed_annot)
#                         ), 
#                         ), cmap = plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()