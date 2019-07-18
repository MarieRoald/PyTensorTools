import sys
sys.path.append('../PyTensor')
sys.path.append('../PyTensor_classification')

from pathlib import Path
import h5py
from scipy.io import savemat
from ..utils import load_summary, open_run, load_best_group


def export_components(experiment_path, run, out_name='factors.mat'):
    experiment_path = Path(experiment_path)
    with open_run(experiment_path, run) as h5:
        checkpoint_group = load_best_group(h5)
        factors = {name: dataset[...] for name, dataset in checkpoint_group.items()}
        attrs = {name: val for name, val in checkpoint_group.attrs.items()}
    
    filetype = out_name.split('.')[-1].lower()
    if filetype == 'mat':
        savemat(experiment_path/out_name, factors)
    elif filetype == 'h5' or filetype == 'hdf5':
        with h5py.File(experiment_path/out_name, 'w') as h5:
            for name, value in factors.items():
                h5[name] = value
                for name, val in attrs.items():
                    h5.attrs[name] = val
    else:
        raise ValueError(f'Invalid file type "{filetype}"')


def export_best_components(experiment_path, out_name='factors.mat'):
    best_run = load_summary(experiment_path)['best_run']
    export_components(experiment_path, best_run, out_name=out_name)

