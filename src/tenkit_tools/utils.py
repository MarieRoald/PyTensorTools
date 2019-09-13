import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from scipy.stats import ttest_ind

import tenkit.utils

from . import preprocessor

from functools import partial

@contextmanager
def open_run(experiment_path, run, mode="r"):
    run_path = Path(experiment_path) / "checkpoints" / run
    h5 = h5py.File(run_path, mode)
    yield h5.__enter__()
    h5.__exit__(None, None, None)


@contextmanager
def open_best_run(experiment_path, mode="r"):
    best_run = load_summary(experiment_path)["best_run"]
    ctx = open_run(experiment_path, best_run, mode=mode)
    yield ctx.__enter__()
    ctx.__exit__(None, None, None)


def load_best_group(run_h5):
    final_it = run_h5.attrs["final_iteration"]
    return run_h5[f"checkpoint_{final_it:05d}"]


def load_summary(experiment_path):
    experiment_path = Path(experiment_path)
    with (experiment_path / "summaries" / "summary.json").open() as f:
        return json.load(f)


def load_evaluations(experiment_path):
    experiment_path = Path(experiment_path)
    with (experiment_path / "summaries" / "evaluations.json").open() as f:
        return json.load(f)


def data_driven_get_sign(factor_matrix, data_matrix):
    return tenkit.utils.get_signs(factor_matrix, data_matrix)[0].reshape([1, -1])


def sign_driven_get_sign(factor_matrix):
    return tenkit.utils.get_signs(factor_matrix, None)[0].reshape([1, -1])


def classification_driven_get_sign(
    factor_matrix, labels, positive_label_value=None, separation_factor_matrix=None
):
    if separation_factor_matrix is None:
        separation_factor_matrix = factor_matrix
    if positive_label_value is not None:
        labels = labels == positive_label_value

    # For each component
    # Find the t-statistic
    # Flip component if t-statistic is negative

    positive = separation_factor_matrix[labels]
    negative = separation_factor_matrix[~labels]

    t_statistics, _ = ttest_ind(positive, negative, axis=0, equal_var=False)
    signs = np.sign(t_statistics)

    return signs.reshape([1, -1])


def load_experiment_params(experiment_path):
    experiment_path = Path(experiment_path)
    parameters_path = experiment_path/"parameters"
    if not parameters_path.is_dir():
        raise RuntimeError(
            f"{experiment_path} is not the path to an experiment. The \"parameters\" folder is missing."
        )

    params = {}
    for parameter in ["data_reader", "experiment", "decomposition", "log"]:
        parameter = f"{parameter}_params"
        parameter_file = parameters_path / f"{parameter}.json"

        with parameter_file.open() as f:
            params[parameter] = json.load(f)
    
    preprocessor_file = parameters_path/"preprocessor_params.json"
    if preprocessor_file.is_file():
        with preprocessor_file.open() as f:
            params["preprocessor_params"] = json.load(f)
    else:
        params["preprocessor_params"] = None
    
    return params


class TestDefaults:
    def __init__(self):
        self._defaults = {}

    def __getitem__(self, item):
        return self._defaults[item]

    def __delitem__(self, item):
        del self._defaults[item]

    def __setitem__(self, key, value):
        self._defaults[key] = value

    def __contains__(self, value):
        return value in self._defaults

    def __iter__(self):
        return iter(self._defaults)

    def items(self):
	    return self._defaults.items()

    def keys(self):
        return self._defaults.keys()

    def values(self):
        return self._defaults.values()

    def set_default(self, defaults):
        def set_default_(cls):
            self[cls] = defaults
            return cls
        
        return set_default_