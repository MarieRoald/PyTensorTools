from abc import ABC, abstractmethod
from pathlib import Path
from operator import itemgetter
import numpy as np

import h5py

from .base_evaluator import BaseEvaluator, create_evaluators
import pytensor


def _sort_by(l, sort_by):
    return [i for _, i in sorted(zip(sort_by, l), key=itemgetter(0))]


class BaseMultipleEvaluator(BaseEvaluator):
    def __init__(self, summary, runs=None):
        super().__init__(summary)
        self.runs=runs

    def __call__(self, data_reader, checkpoint_path):
        """Returns a dict whose keys are column names and values are result lists
        """
        return self._evaluate(data_reader, checkpoint_path)
    
    @abstractmethod
    def _evaluate(self, data_reader, checkpoint_path):
        pass
    
    def checkpoint_files(self, checkpoint_path):
        if self.runs is None:
            return sorted(Path(checkpoint_path).glob('run_*.h5'))
        else:
            return [checkpoint_path/run for run in self.runs]
    
    def load_final_checkpoints(self, checkpoint_path):
        """Generator that yields the final checkpoint for all runs.
        """
        for checkpoint in self.checkpoint_files(checkpoint_path):
            with h5py.File(checkpoint, 'r') as h5:
                decomposition = self.load_final_checkpoint(h5)
            yield checkpoint.name, decomposition


class MultipleSingleRunEvaluators(BaseMultipleEvaluator):
    def __init__(self, summary, single_run_evaluator_params, runs):
        super().__init__(summary, runs)
        self.single_run_evaluator = create_evaluators(single_run_evaluator_params, summary)
        self._name = f'Multiple {self.single_run_evaluator.name}'
    

    def _evaluate(self, data_reader, checkpoint_path):
        results = {'run': [], self.single_run_evaluator.name: []}
        for checkpoint in self.checkpoint_files(checkpoint_path):
            with h5py.File(checkpoint) as h5:
                results['run'].append(checkpoint.name)
                results[self.single_run_evaluator.name].append(self.single_run_evaluator(data_reader, h5))
        
        return results

class Uniqueness(BaseMultipleEvaluator):
    """Note: Bases similarity on runs on SSE. """
    def _get_best_run(self, checkpoint_path):
        best_run_path = checkpoint_path/self.summary['best_run']
        with h5py.File(best_run_path) as best_run:
            decomposition = self.load_final_checkpoint(best_run)
        return decomposition

    def _SSE_difference(self, decomposition1, decomposition2):
        return np.sum(decomposition1.construct_tensor() - decomposition2.construct_tensor())

    def _factor_match_score(self, decomposition1, decomposition2):
        return pytensor.metrics.factor_match_score(decomposition1, decomposition2)[0]

    def _evaluate(self, data_reader, checkpoint_path):
        best_decomposition = self._get_best_run(checkpoint_path)
        results = {'name': [], 'SSE_difference': [], 'fms': []}

        for name, decomposition in self.load_final_checkpoints(checkpoint_path):
            results['name'].append(name)
            results['SSE_difference'].append(self._SSE_difference(best_decomposition, decomposition))
            results['fms'].append(self._factor_match_score(best_decomposition.factor_matrices, decomposition.factor_matrices))

        results['name'] = _sort_by(results['name'], results['SSE_difference'])
        results['fms'] = _sort_by(results['fms'], results['SSE_difference'])
        results['SSE_difference'] = sorted(results['SSE_difference'])
        
        return results