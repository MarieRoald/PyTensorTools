import multiprocessing
from abc import ABC, abstractproperty, abstractmethod

from . import datareader
from pytensor.decomposition import cp
from pytensor.decomposition import parafac2


class BaseExperiment(ABC):
    def __init__(self, experiment_params, data_reader_params, decomposition_params, log_params):
        self.experiment_params = experiment_params
        self.data_reader_params = data_reader_params
        self.decomposition_params = decomposition_params
        self.log_params = log_params

        self.data_reader = self.generate_data_reader()
    
    def generate_data_reader(self):
        DataReader = getattr(datareader, self.data_reader_params['type'])
        return DataReader(**self.data_reader_params['arguments'])

    @abstractmethod
    def generate_decomposer(self):
        pass
    
    def run_single_experiment(self, _=None):
        decomposer = self.generate_decomposer()
        X = self.data_reader.tensor
        decomposer.fit(X, **self.decomposition_params.get('fit_params', {}))
        return decomposer
    
    def run_many_experiments(self, num_experiments):
        if 'num_processess' in self.experiment_params:
            num_processess = self.experiment_params['num_processess']
        else:
            num_processess = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(num_processess) as p:
            return p.map(self.run_single_experiment, range(num_experiments))
        

class CPExperiment(BaseExperiment):
    def generate_decomposer(self):
        Decomposer = getattr(cp, self.decomposition_params['type'])
        return Decomposer(**self.decomposition_params['arguments'])


class Parafac2Experiment(BaseExperiment):
    def generate_decomposer(self):
        Decomposer = getattr(parafac2, self.decomposition_params['type'])
        return Decomposer(**self.decomposition_params['arguments'])

