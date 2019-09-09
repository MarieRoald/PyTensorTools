from abc import ABC, abstractmethod
from pathlib import Path
from operator import itemgetter
import numpy as np

import h5py

from .base_evaluator import BaseEvaluator, create_evaluator
import tenkit


def _sort_by(l, sort_by):
    return [i for _, i in sorted(zip(sort_by, l), key=itemgetter(0))]


class BaseMultipleEvaluator(BaseEvaluator):
    def __init__(self, summary, runs=None, **kwargs):
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
    def __init__(self, summary, single_run_evaluator_params, runs, **kwargs):
        super().__init__(summary, runs)
        self.single_run_evaluator = create_evaluator(single_run_evaluator_params, summary)
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

    def _SSE(self, data_reader, decomposition):
        return np.sum((decomposition.construct_tensor() - data_reader.tensor)**2)

    def _SSE_difference(self, data_reader, decomposition1, decomposition2):
        SSE1 = self._SSE(data_reader, decomposition1)
        SSE2 = self._SSE(data_reader, decomposition2)
        return SSE2 - SSE1
    
    def _fit_difference(self, data_reader, decomposition1, decomposition2):
        Xvar = np.sum(data_reader.tensor**2)
        SSE1 = self._SSE(data_reader, decomposition1)
        fit1 = 1 - SSE1/Xvar
        SSE2 = self._SSE(data_reader, decomposition2)
        fit2 = 1 - SSE2/Xvar

        return fit1 - fit2

    def _factor_match_score(self, decomposition1, decomposition2):
        return tenkit.metrics.factor_match_score(decomposition1, decomposition2)[0]

    def _evaluate(self, data_reader, checkpoint_path):
        best_decomposition = self._get_best_run(checkpoint_path)
        results = {'name': [], 'SSE_difference': [], 'fit_difference': [], 'fms': []}

        # Do in parallel.
        for name, decomposition in self.load_final_checkpoints(checkpoint_path):
            results['name'].append(name)
            results['SSE_difference'].append(self._SSE_difference(data_reader, best_decomposition, decomposition))
            results['fit_difference'].append(self._fit_difference(data_reader, best_decomposition, decomposition))
            results['fms'].append(self._factor_match_score(best_decomposition.factor_matrices, decomposition.factor_matrices))

        results['name'] = _sort_by(results['name'], results['SSE_difference'])
        results['fms'] = _sort_by(results['fms'], results['SSE_difference'])
        results['fit_difference'] = _sort_by(results['fit_difference'], results['SSE_difference'])
        results['SSE_difference'] = sorted(results['SSE_difference'])
        
        return results

class Parafac2Uniqueness(Uniqueness):
    def _factor_match_score(self, decomposition1, decomposition2):
        return tenkit.metrics.factor_match_score_parafac2(decomposition1, decomposition2)[0]
