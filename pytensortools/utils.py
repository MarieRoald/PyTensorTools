import json
from pathlib import Path
from contextlib import contextmanager
import h5py


@contextmanager
def open_run(experiment_path, run, mode='r'):
    run_path = Path(experiment_path)/'checkpoints'/run
    h5 = h5py.File(run_path, mode)
    yield h5.__enter__()
    h5.__exit__()


@contextmanager
def open_best_run(experiment_path, mode='r'):
    best_run = load_summary(experiment_path)['best_run']
    ctx = open_run(experiment_path, best_run, mode=mode)
    yield ctx.__enter__()
    ctx.__exit__()


def load_best_group(run_h5):
    final_it = run_h5.attrs['final_iteration']
    return run_h5[f'checkpoint_{final_it:05d}']


def load_summary(experiment_path):
    with (experiment_path/'summaries'/'summary.json').open() as f:
        return json.load(f)
    
def load_evaluations(experiment_path):
    with (experiment_path/'summaries'/'evaluations.json').open() as f:
        return json.load(f)

