from sympy import E
from utils                      import path_gen


# Data
path                = '/mnt/sdg/maxs'
annot_type          = 'soma'

# Training
data_set            = 'LIVECell'
data_type           = 'per_celltype'
data_subset         = 'BV2'    
subset_type         = 'test'
n_epochs            = 200

counter = -1
ap_dict = {}
afnr_dict = {}
af1_dict = {}

network_names = ['D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W']
for duplicate in range(3):
    for iN, network_name in enumerate(network_names):
        save_model_name=f'{network_names[iN]}{duplicate}'

        results_folder = path_gen([
            path,
            'results',
            data_set,
            data_type,
            data_subset,
            subset_type,
            annot_type,
            save_model_name,
            'deployment'
        ])

        try:
            print(f'{results_folder}checkpoint{n_epochs}_performance_metrics.txt')
            with open(f'{results_folder}checkpoint{n_epochs}_performance_metrics.txt', 'r') as infile:
                counter = counter + 1
                for line in infile:
                    if line.startswith('ap['):
                        line = line[3:-2]
                        ap = [s for s in line.split()]
                    if line.startswith('afnr['):
                        line = line[5:-2]
                        afnr = [s for s in line.split()]
                    if line.startswith('af1['):
                        line = line[4:-1]
                        af1 = [s for s in line.split()]
                ap_dict[save_model_name] = ap[0]
                afnr_dict[save_model_name] = afnr[0]
                af1_dict[save_model_name] = af1[0]
        except:
            pass

a=1