import multiprocessing
from abc import ABC, abstractproperty, abstractmethod

from . import datareader
from pytensor.decomposition import cp
from pytensor.decomposition import parafac2
import pytensor
from pathlib import Path
import json

import numpy as np
class Experiment(ABC):
    def __init__(self, experiment_params, data_reader_params, decomposition_params, log_params):
        self.experiment_params = experiment_params
        self.data_reader_params = data_reader_params
        self.decomposition_params = decomposition_params
        self.log_params = log_params 
        self.data_reader = self.generate_data_reader()
       
        self.create_experiment_directories()

    def create_experiment_directories(self):
        experiment_path = Path(self.experiment_params['save_path'])

        self.checkpoint_path = experiment_path / 'checkpoints'
        self.parameter_path = experiment_path / 'parameters'
        self.summary_path = experiment_path / 'summaries'

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

    def get_experiment_statistics(self):
        # TODO: This can load the list of decompositions
        model_type = getattr(pytensor.decomposition, self.decomposition_params['type'])
        model_rank = self.decomposition_params['arguments']['rank']

        best_run = ''
        best_fit = -1

        losses = []
        fits = []

        for file_name in self.checkpoint_path.glob('run*.h5'):
            decomposer = model_type(rank=model_rank, init_scheme='from_checkpoint')
            decomposer._init_fit(self.data_reader.tensor, initial_decomposition=file_name)

            losses.append(decomposer.loss())
            fits.append(decomposer.fit)
            if decomposer.fit > best_fit:
                best_run = file_name
                best_fit = decomposer.fit
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

        self.summary['dataset_path'] = self.data_reader['arguments']['file_path']
        self.summary['dataset_path'] = self.data_reader['arguments']['file_path']

        self.summary = {**self.summary, **self.get_experiment_statistics()}        # finne beste run

        return self.summary
    
    def save_summary(self):
        summary = self.create_summary()
        summary_path = self.summary_path / 'summary.json'

        with summary_path.open('w') as f:
            json.dump(summary, f)

    def generate_data_reader(self):
        DataReader = getattr(datareader, self.data_reader_params['type'])
        return DataReader(**self.data_reader_params['arguments'])

    def generate_loggers(self):
        loggers = []
        for logger_params in self.log_params:
            Logger = getattr(pytensor.decomposition.logging, logger_params['type'])
            loggers.append(Logger(**logger_params.get('arguments', {})))

        return loggers

    def generate_decomposer(self, checkpoint_path=None):
        Decomposer = getattr(pytensor.decomposition, self.decomposition_params['type'])
        return Decomposer(**self.decomposition_params['arguments'], loggers=self.generate_loggers(), checkpoint_path=checkpoint_path)
    
    def run_single_experiment(self, run_num=None):
        checkpoint_path = None

        if run_num is not None:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.is_dir():
                checkpoint_path.mkdir(parents=True)
            checkpoint_path = str(checkpoint_path/f'run_{run_num}.h5')

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
            return p.map(self.run_single_experiment, range(num_experiments))

    

    def run_experiments(self):
        self.copy_parameter_files()
        decomposers = self.run_many_experiments(self.experiment_params.get('num_runs', 10))
        self.save_summary()

        return decomposers
        

