from abc import ABC, abstractmethod
from scipy.stats import ttest_ind
import pytensor
from pytensor import metrics  # TODO: Fix __init__.py
import itertools
import numpy as np

class BaseSingleRunEvaluator(ABC):
    _name = None
    def __call__(self, data_reader, h5):
        return self._evaluate(data_reader, h5)

    @abstractmethod
    def _evaluate(self, data_reader, h5):
        pass

    @property
    def name(self):
        if self._name is None:
            return type(self).__name__
        return self._name


class FinalLoss(BaseSingleRunEvaluator):
    _name = 'Final loss'
    def _evaluate(self, data_reader, h5):
        return h5['LossLogger/values'][-1]

class ExplainedVariance(BaseSingleRunEvaluator):
    #TODO: maybe create a decomposer to not rely on logging
    _name = 'Explained variance'
    def _evaluate(self, data_reader, h5):
        return h5['ExplainedVarianceLogger/values'][-1]

class PValue(BaseSingleRunEvaluator):
    _name = 'Best P value'
    def __init__(self, mode, decomposer_type):
        self.mode = mode
        self._name = f'Best P value for mode {mode}'
        decomposition_type = getattr(pytensor.decomposition,decomposer_type).DecompositionType
        self.decomposition_type = decomposition_type

    def _evaluate(self, data_reader, h5):
        final_it = h5.attrs['final_iteration']
        decomposition = self.decomposition_type.load_from_hdf5_group(h5[f'checkpoint_{final_it:05d}'])
        factors = decomposition.factor_matrices[self.mode]

        classes = data_reader.classes.squeeze()

        assert len(set(classes)) == 2

        indices = [[i for i, c in enumerate(classes) if c == class_] for class_ in set(classes)]
        p_values = tuple(ttest_ind(factors[indices[0]], factors[indices[1]], equal_var=False).pvalue)
        return min(p_values)

class WorstDegeneracy(BaseSingleRunEvaluator):
    _name = 'Worst degeneracy'
    def __init__(self, decomposer_type, modes=None):
        self.decomposition_type = getattr(pytensor.decomposition,decomposer_type).DecompositionType
        self.modes = modes

    def _evaluate(self, data_reader, h5):
        #TODO: abstrahere vekk?
        final_it = h5.attrs['final_iteration']
        decomposition = self.decomposition_type.load_from_hdf5_group(h5[f'checkpoint_{final_it:05d}'])
        factors = decomposition.factor_matrices
        if self.modes is None:
            modes = range(len(decomposition.factor_matrices))
        # A, B, C = factors
        # R = A.shape[1]

        # assert A.shape[1] == B.shape[1] == C.shape[1]

        R = decomposition.factor_matrices[0].shape[1]
        min_score = np.inf
        for (p1,p2) in itertools.permutations(range(R), r=2):
            factors_p1 = [fm[:, p1] for mode, fm in enumerate(decomposition.factor_matrices) if mode in modes]
            factors_p2 = [fm[:, p2] for mode, fm in enumerate(decomposition.factor_matrices) if mode in modes]

            score = metrics._factor_match_score(factors_p1, factors_p2,
                                                 nonnegative=False, weight_penalty=False)[0]
            # score = metrics._factor_match_score([A[:,p1], B[:,p1], C[:, p1]], [A[:,p2], B[:,p2], C[:, p2]], 
            #                                      nonnegative=False, weight_penalty=False)[0]

            if score<min_score:
                min_score=score

        return min_score
