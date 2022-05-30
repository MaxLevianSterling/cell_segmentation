from eval import binary2json  
from eval import evaluate


network_names = ['D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W']

for duplicate in range(3):
    for iN, network_name in enumerate(network_names):
        binary2json(
            path = '/mnt/sdg/maxs',
            data_set = 'LIVECell',
            data_type = 'part_set',
            data_subset = '5',
            subset_type = 'test',
            annot_type = 'soma',
            mode = 'eval',
            model=f'{network_names[iN]}{duplicate}',
            checkpoint = 1000,
            print_separator = '$',
        )
        #try:
        evaluate(
            path = '/mnt/sdg/maxs',
            data_set = 'LIVECell',
            data_type = 'part_set',
            data_subset = '5',
            subset_type = 'test',
            annot_type = 'soma',
            mode = 'original',
            model=f'{network_names[iN]}{duplicate}',
            checkpoint = 1000,
            print_separator = '$',
        )
        #except:
        #    pass