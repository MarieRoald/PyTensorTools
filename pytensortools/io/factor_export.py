from pathlib import Path
import h5py
from scipy.io import savemat
from ..utils import load_summary, open_run, load_best_group


def export_components(experiment_path, run, out_name='factors.mat'):
    experiment_path = Path(experiment_path)
    with open_run(experiment_path, run) as h5:
        checkpoint_group = load_best_group(h5)
        factors = {name: dataset[...] for name, dataset in checkpoint_group.items()}
    savemat(experiment_path/out_name, factors)


def export_best_components(experiment_path, out_name='factors.mat'):
    best_run = load_summary(experiment_path)['best_run']
    export_components(experiment_path, best_run, out_name=out_name)
