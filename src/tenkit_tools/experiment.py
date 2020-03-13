import json
import multiprocessing
import pickle
from abc import ABC, abstractmethod, abstractproperty
from copy import copy
from functools import partial
from pathlib import Path
from time import sleep
from typing import Dict

import numpy as np

import tenkit
from tenkit.decomposition import cp, parafac2

from . import datareader, preprocessor

EXPERIMENT_COMPLETED = 0
EXPERIMENT_INTERRUPTED = 1
DEBUG = False


def _raise_multiprocessing_error(run_id):
    def raise_exception(exception):
        print(f"Exception for run {run_id}")
        raise exception


def generate_data_reader(data_reader_params):
    DataReader = getattr(datareader, data_reader_params["type"])
    args = data_reader_params.get("arguments", {})
    return DataReader(**args)


def generate_loggers(log_params):
    loggers = []
    for logger_params in log_params:
        Logger = getattr(tenkit.decomposition.logging, logger_params["type"])
        loggers.append(Logger(**logger_params.get("arguments", {})))

    return loggers


def generate_decomposer(
    decomposition_params, logger_params, checkpoint_path, run_num, load_old
):
    if not checkpoint_path.is_dir():
        checkpoint_path.mkdir(parents=True)
    checkpoint_path = str(checkpoint_path / f"run_{run_num:03d}.h5")

    kwargs = copy(decomposition_params.get("arguments", {}))
    if load_old:
        kwargs["init"] = str(checkpoint_path)

    Decomposer = getattr(tenkit.decomposition, decomposition_params["type"])
    return Decomposer(
        **kwargs,
        loggers=generate_loggers(logger_params),
        checkpoint_path=checkpoint_path,
    )


def preprocess_data(data_reader, preprocessors_params):
    if isinstance(preprocessors_params, Dict):
        preprocessor_params = [preprocessors_params]

    preprocessed = data_reader
    for preprocessor_params in preprocessors_params:
        Preprocessor = getattr(preprocessor, preprocessor_params["type"])
        args = preprocessor_params.get("arguments", {})
        preprocessed = Preprocessor(data_reader=preprocessed, **args)

    return preprocessed


def run_partial_experiment(
    decomposition_params,
    log_params,
    data_reader_params,
    preprocessors_params,
    checkpoint_path,
    run_num,
    load_old,
    seed=None,
):
    np.random.seed(seed)
    data_reader = generate_data_reader(data_reader_params)
    data_reader = preprocess_data(data_reader, preprocessors_params)
    decomposer = generate_decomposer(
        decomposition_params, log_params, checkpoint_path, run_num, load_old
    )
    X = data_reader.tensor

    fit_params = decomposition_params.get("fit_params", {})
    decomposer.fit(X, **fit_params)


class Experiment(ABC):
    def __init__(
        self,
        experiment_params,
        data_reader_params,
        decomposition_params,
        log_params,
        preprocessor_params=None,
        load_id=None,
    ):
        # Set params dicts
        self.experiment_params = experiment_params
        self.data_reader_params = data_reader_params
        self.preprocessor_params = preprocessor_params
        self.decomposition_params = decomposition_params
        self.log_params = log_params
        self.data_reader = self.generate_data_reader()
        if self.preprocessor_params is not None:
            self.data_reader = self.preprocess_data(self.data_reader)

        # Check if we should load from checkpoint
        self.load_id = load_id
        self.load_old = False
        if load_id is not None:
            self.load_old = True

        # Set experiment paths
        self.experiment_path = self.get_experiment_directory()
        self.create_experiment_directories()

    @property
    def num_processes(self):
        # TODO: Skrivefeil her?
        if "num_processess" in self.experiment_params:
            return self.experiment_params["num_processess"]
        else:
            return multiprocessing.cpu_count() - 1

    def get_experiment_directory(self):
        # For easy access, create variables from dict
        save_path = Path(self.experiment_params["save_path"])
        experiment_name = self.experiment_params["experiment_name"]

        # Set parent dir and experiment name
        parent = save_path / experiment_name
        name = f'{experiment_name}_rank_{self.decomposition_params["arguments"]["rank"]:02d}'

        # If loading old, return correct path
        if self.load_id is not None:
            return Path(f"{parent/name}_{self.load_id:02d}")

        # Give unique folder to current experiment
        num = 0
        experiment_path = Path(f"{parent/name}_{num:02d}")
        while experiment_path.is_dir():
            num += 1
            experiment_path = Path(f"{parent/name}_{num:02d}")

        return experiment_path

    def create_experiment_directories(self):
        self.checkpoint_path = self.experiment_path / "checkpoints"
        self.parameter_path = self.experiment_path / "parameters"
        self.summary_path = self.experiment_path / "summaries"

        for path in [self.checkpoint_path, self.parameter_path, self.summary_path]:
            if not Path.is_dir(path):
                path.mkdir(parents=True)

    def copy_parameter_files(self):
        try:
            json.dumps(self.experiment_params)
            json.dumps(self.data_reader_params)
            json.dumps(self.decomposition_params)
            json.dumps(self.log_params)
            if self.preprocessor_params is not None:
                json.dumps(self.preprocessor_params)

        except TypeError:
            with (self.parameter_path / "experiment_params.pickle").open("wb") as f:
                pickle.dump(self.experiment_params, f)
    
            with (self.parameter_path / "data_reader_params.pickle").open("wb") as f:
                pickle.dump(self.data_reader_params, f)
    
            with (self.parameter_path / "decomposition_params.pickle").open("wb") as f:
                pickle.dump(self.decomposition_params, f)
    
            with (self.parameter_path / "log_params.pickle").open("wb") as f:
                pickle.dump(self.log_params, f)
    
            if self.preprocessor_params is not None:
                with (self.parameter_path / "preprocessor_params.pickle").open("wb") as f:
                    pickle.dump(self.preprocessor_params, f)

            return


        with (self.parameter_path / "experiment_params.json").open("w") as f:
            json.dump(self.experiment_params, f)

        with (self.parameter_path / "data_reader_params.json").open("w") as f:
            json.dump(self.data_reader_params, f)

        with (self.parameter_path / "decomposition_params.json").open("w") as f:
            json.dump(self.decomposition_params, f)

        with (self.parameter_path / "log_params.json").open("w") as f:
            json.dump(self.log_params, f)

        if self.preprocessor_params is not None:
            with (self.parameter_path / "preprocessor_params.json").open("w") as f:
                json.dump(self.preprocessor_params, f)

    def get_experiment_statistics(self):
        # TODO: This can load the list of decompositions
        model_type = getattr(tenkit.decomposition, self.decomposition_params["type"])
        model_rank = self.decomposition_params["arguments"]["rank"]

        best_run = ""
        best_fit = -np.inf  # endret fra -1

        losses = []
        fits = []

        for file_name in self.checkpoint_path.glob("run*.h5"):
            decomposer = model_type(**self.decomposition_params["arguments"])
            decomposer.init = "from_checkpoint"
            decomposer._init_fit(
                self.data_reader.tensor, max_its=None, initial_decomposition=file_name,
            )

            losses.append(decomposer.loss)
            fits.append(decomposer.explained_variance)
            if decomposer.explained_variance > best_fit:
                best_run = file_name.name
                best_fit = decomposer.explained_variance
                best_loss = decomposer.loss

        std_loss = np.std(losses)
        std_fit = np.std(fits)

        return {
            "best_run": best_run,
            "best_fit": best_fit,
            "best_loss": best_loss,
            "std_loss": std_loss,
            "std_fit": std_fit,
        }

    def create_summary(self, completion_status):
        self.summary = {}

        self.summary["dataset_path"] = self.data_reader_params["arguments"]["file_path"]
        self.summary["model_type"] = self.decomposition_params["type"]
        self.summary["model_rank"] = self.decomposition_params["arguments"]["rank"]
        self.summary["dataset_shape"] = self.generate_data_reader().tensor.shape
        self.summary["experiment_completed"] = completion_status == EXPERIMENT_COMPLETED

        self.summary = {
            **self.summary,
            **self.get_experiment_statistics(),
        }  # finne beste run

        return self.summary

    def save_summary(self, completion_status):
        summary = self.create_summary(completion_status=completion_status)
        summary_path = self.summary_path / "summary.json"

        with summary_path.open("w") as f:
            json.dump(summary, f)
	

    def generate_data_reader(self):
        DataReader = getattr(datareader, self.data_reader_params["type"])
        args = self.data_reader_params.get("arguments", {})
        return DataReader(**args)

    def preprocess_data(self, data_reader):
        if isinstance(self.preprocessor_params, Dict):
            self.preprocessor_params = [self.preprocessor_params]

        preprocessed = data_reader
        for preprocessor_params in self.preprocessor_params:
            Preprocessor = getattr(preprocessor, preprocessor_params["type"])
            args = preprocessor_params.get("arguments", {})
            preprocessed = Preprocessor(data_reader=preprocessed, **args)

        return preprocessed

    def generate_loggers(self):
        return generate_loggers(self.log_params)

    def print_experiment_info(self):
        data_reader = generate_data_reader(self.data_reader_params)
        data_reader = preprocess_data(data_reader, self.preprocessor_params)
        X = data_reader.tensor

        rank = self.decomposition_params["arguments"]["rank"]
        max_its = self.decomposition_params["arguments"].get("max_its", "Default")
        tol = self.decomposition_params["arguments"].get("tol", "Default")

        print("Starting fit:")
        print(f"  * Tensor shape: {X.shape}")
        print(f"  * Decomposition: {self.decomposition_params['type']}")
        print(f"  * Rank: {rank}")
        print(f"  * Maximum number of iterations: {max_its}")
        print(f"  * Tolerance: {tol}")

    def run_many_experiments(self, num_experiments, should_print):
        if should_print:
            self.print_experiment_info()

        if DEBUG:
            results = [run_partial_experiment(run_num=i, decomposition_params=self.decomposition_params, log_params=self.log_params, data_reader_params=self.data_reader_params, preprocessors_params=self.preprocessor_params, checkpoint_path=self.checkpoint_path, load_old=self.load_old) for i in range(num_experiments)]
            return EXPERIMENT_COMPLETED

        with multiprocessing.Pool(self.num_processes) as pool:
            results = []
            for i in range(num_experiments):
                result = pool.apply_async(
                    run_partial_experiment,
                    kwds={
                        "run_num": i,
                        "decomposition_params": self.decomposition_params,
                        "log_params": self.log_params,
                        "data_reader_params": self.data_reader_params,
                        "preprocessors_params": self.preprocessor_params,
                        "checkpoint_path": self.checkpoint_path,
                        "load_old": self.load_old,
                    },
                    error_callback=_raise_multiprocessing_error(i),
                )
                results.append(result)

            try:
                while True:
                    sleep(0.5)
                    if all(result.ready() for result in results):
                        print("Experiment completed")
                        [result.get() for result in results]
                        break
            except KeyboardInterrupt:
                pool.terminate()
                print("Experiment interrupted")
                return EXPERIMENT_INTERRUPTED
        return EXPERIMENT_COMPLETED

    def run_experiments(self, should_print=True):
        self.copy_parameter_files()
        # Pass på at init er
        completion_status = self.run_many_experiments(
            self.experiment_params.get("num_runs", 10), should_print
        )
        self.save_summary(completion_status=completion_status)
        print(f"Stored summaries in {self.experiment_path}")

    def save_raw_dataset(self, label_names, out_file):
        self.generate_data_reader().to_matlab(label_names, out_file)

    def save_preprocessed_dataset(self, label_names, out_file):
        self.data_reader.to_matlab(label_names, out_file)
