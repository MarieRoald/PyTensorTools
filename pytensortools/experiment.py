import multiprocessing
from abc import ABC, abstractproperty, abstractmethod

from . import datareader
from pytensor.decomposition import cp
from pytensor.decomposition import parafac2
import pytensor
from pathlib import Path

class Experiment(ABC):
    def __init__(self, experiment_params, data_reader_params, decomposition_params, log_params):
        self.experiment_params = experiment_params
        self.data_reader_params = data_reader_params
        self.decomposition_params = decomposition_params
        self.log_params = log_params

        self.data_reader = self.generate_data_reader()
        self.checkpoint_path = experiment_params['save_path']

    
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
        return self.run_many_experiments(self.experiment_params.get('num_runs', 10))
        

