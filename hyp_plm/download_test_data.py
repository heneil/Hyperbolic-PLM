import os
from esm.data import ESMStructuralSplitDataset
ESMStructuralSplitDataset(split_level='superfamily', cv_partition='4', split='train', root_path=os.path.expanduser('/data/neilhe2/esm_data'), download=True)
print('done')