import multiprocessing
from abc import ABC, abstractproperty, abstractmethod
import json
from typing import Dict

from . import datareader
from . import preprocessor
from pytensor.decomposition import cp
from pytensor.decomposition import parafac2
import pytensor
from pathlib import Path

from functools import partial

import numpy as np


def generate_data_reader(data_reader_params):
    DataReader = getattr(datareader, data_reader_params['type'])
    args = data_reader_params.get('arguments', {})
    return DataReader(**args)


def generate_loggers(log_params):
    loggers = []
    for logger_params in log_params:
        Logger = getattr(pytensor.decomposition.logging, logger_params['type'])
        loggers.append(Logger(**logger_params.get('arguments', {})))

    return loggers


def generate_decomposer(decomposition_params, logger_params, checkpoint_path, run_num):
    if not checkpoint_path.is_dir():
        checkpoint_path.mkdir(parents=True)
    checkpoint_path = str(checkpoint_path/f'run_{run_num:03d}.h5')

    Decomposer = getattr(pytensor.decomposition, decomposition_params['type'])
    return Decomposer(
        **decomposition_params.get('arguments', {}),
        loggers=generate_loggers(logger_params),
        checkpoint_path=checkpoint_path
    )


def preprocess_data(data_reader, preprocessors_params):
    if isinstance(preprocessors_params, Dict):
        self.preprocessor_params = [preprocessors_params]
    
    preprocessed = data_reader
    for preprocessor_params in preprocessors_params:
        Preprocessor = getattr(preprocessor, preprocessor_params['type'])
        args = preprocessor_params.get('arguments', {})
        preprocessed = Preprocessor(data_reader=preprocessed, **args)
    
    return preprocessed



def run_partial_experiment(
    decomposition_params,
    logger_params,
    data_reader_params,
    preprocessors_params,
    checkpoint_path,
    run_num,
    seed=None
):
    np.random.seed(seed)
    data_reader = generate_data_reader(data_reader_params)
    data_reader = preprocess_data(data_reader, preprocessors_params)
    decomposer = generate_decomposer(decomposition_params, logger_params, checkpoint_path, run_num)

    X = data_reader.tensor
    fit_params = decomposition_params.get('fit_params', {})
    decomposer.fit(X, **fit_params)


class Experiment(ABC):
    def __init__(self, experiment_params, data_reader_params, decomposition_params, log_params, preprocessor_params=None):
        self.experiment_params = experiment_params
        self.data_reader_params = data_reader_params
        self.preprocessor_params = preprocessor_params
        self.decomposition_params = decomposition_params
        self.log_params = log_params 
        self.data_reader = self.generate_data_reader()
        if self.preprocessor_params is not None:
            self.data_reader = self.preprocess_data(self.data_reader)
       
        self.experiment_path = self.get_experiment_directory()
        self.create_experiment_directories()

    def get_experiment_directory(self):
        # For easy access, create variables from dict
        save_path = Path(self.experiment_params["save_path"])
        experiment_name = self.experiment_params["experiment_name"]

        # Set parent dir and experiment name        
        parent = save_path/experiment_name
        name = f'{experiment_name}_rank_{self.decomposition_params["arguments"]["rank"]:02d}'

        # Give unique folder to current experiment
        num = 0
        experiment_path = Path(f'{parent/name}_{num:02d}')
        while experiment_path.is_dir():
            num += 1
            experiment_path = Path(f'{parent/name}_{num:02d}')
        
        return experiment_path

    def create_experiment_directories(self):
        self.checkpoint_path = self.experiment_path / 'checkpoints'
        self.parameter_path = self.experiment_path / 'parameters'
        self.summary_path = self.experiment_path / 'summaries'

        for path in [self.checkpoint_path, self.parameter_path, self.summary_path]:
            if not Path.is_dir(path):
                path.mkdir(parents=True)

    def copy_parameter_files(self):
        with (self.parameter_path / 'experiment_params.json').open('w') as f:
            json.dump(self.experiment_params, f)

        with (self.parameter_path / 'data_reader_params.json').open('w') as f:
            json.dump(self.data_reader_params, f)

        with (self.parameter_path / 'decomposition_params.json').open('w') as f:
            json.dump(self.decomposition_params, f)

        with (self.parameter_path / 'log_params.json').open('w') as f:
            json.dump(self.log_params, f)
        
        if self.preprocessor_params is not None:
            with (self.parameter_path / 'preprocessor_params.json').open('w') as f:
                json.dump(self.preprocessor_params, f)
            
    def get_experiment_statistics(self):
        # TODO: This can load the list of decompositions
        model_type = getattr(pytensor.decomposition, self.decomposition_params['type'])
        model_rank = self.decomposition_params['arguments']['rank']

        best_run = ''
        best_fit = -np.inf # endret fra -1

        losses = []
        fits = []

        for file_name in self.checkpoint_path.glob('run*.h5'):
            decomposer = model_type(rank=model_rank, max_its=-1, init='from_checkpoint')
            decomposer._init_fit(self.data_reader.tensor, max_its=None, initial_decomposition=file_name)

            losses.append(decomposer.loss())
            fits.append(decomposer.explained_variance)
            if decomposer.explained_variance > best_fit:
                best_run = file_name.name
                best_fit = decomposer.explained_variance
                best_loss = decomposer.loss()

        std_loss = np.std(losses)
        std_fit = np.std(fits)

        return {
            'best_run': best_run,
            'best_fit': best_fit,
            'best_loss': best_loss,
            'std_loss': std_loss,
            'std_fit': std_fit
        }

    def create_summary(self):
        self.summary = {}

        self.summary['dataset_path'] = self.data_reader_params['arguments']['file_path']
        self.summary['model_type'] = self.decomposition_params['type']
        self.summary['model_rank'] = self.decomposition_params['arguments']['rank']

        self.summary = {**self.summary, **self.get_experiment_statistics()}        # finne beste run

        return self.summary
    
    def save_summary(self):
        summary = self.create_summary()
        summary_path = self.summary_path / 'summary.json'

        with summary_path.open('w') as f:
            json.dump(summary, f)

    def generate_data_reader(self):
        DataReader = getattr(datareader, self.data_reader_params['type'])
        args = self.data_reader_params.get('arguments', {})
        return DataReader(**args)

    def preprocess_data(self, data_reader):
        if isinstance(self.preprocessor_params, Dict):
            self.preprocessor_params = [self.preprocessor_params]
        
        preprocessed = data_reader
        for preprocessor_params in self.preprocessor_params:
            Preprocessor = getattr(preprocessor, preprocessor_params['type'])
            args = preprocessor_params.get('arguments', {})
            preprocessed = Preprocessor(data_reader=preprocessed, **args)
        
        return preprocessed

    def generate_loggers(self):
        loggers = []
        for logger_params in self.log_params:
            Logger = getattr(pytensor.decomposition.logging, logger_params['type'])
            loggers.append(Logger(**logger_params.get('arguments', {})))

        return loggers

    def generate_decomposer(self, checkpoint_path=None):
        Decomposer = getattr(pytensor.decomposition, self.decomposition_params['type'])
        return Decomposer(
            **self.decomposition_params['arguments'],
            loggers=self.generate_loggers(),
            checkpoint_path=checkpoint_path
        )
    
    def run_single_experiment(self, run_num=None, seed=None):
        np.random.seed(seed)
        checkpoint_path = None

        if run_num is not None:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.is_dir():
                checkpoint_path.mkdir(parents=True)
            checkpoint_path = str(checkpoint_path/f'run_{run_num:03d}.h5')

        decomposer = self.generate_decomposer(checkpoint_path)
        X = self.data_reader.tensor
        decomposer.fit(X, **self.decomposition_params.get('fit_params', {}))
        return decomposer
    
    def run_many_experiments(self, num_experiments):
        if 'num_processess' in self.experiment_params:
            num_processess = self.experiment_params['num_processess']
        else:
            num_processess = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(num_processess) as p:
            # return [self.run_single_experiment(i) for i in range(num_experiments)]
            run_single_experiment = partial(
                run_partial_experiment,
                self.decomposition_params,
                self.log_params,
                self.data_reader_params,
                self.preprocessor_params,
                self.checkpoint_path,
            )
            #return [run_single_experiment(i) for i in range(num_experiments)]
            return p.map(run_single_experiment, range(num_experiments))

    def run_experiments(self):
        self.copy_parameter_files()
        self.run_many_experiments(self.experiment_params.get('num_runs', 10))
        self.save_summary()
        print(f'Stored summaries in {self.experiment_path}')

        

