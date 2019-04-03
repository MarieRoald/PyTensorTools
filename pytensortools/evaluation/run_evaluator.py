from abc import ABC, abstractmethod
from scipy.stats import ttest_ind
import pytensor

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
    _name = 'Explained Variance'
    def _evaluate(self, data_reader, h5):
        return h5['ExplainedVarianceLogger/values'][-1]

class PValue(BaseSingleRunEvaluator):
    _name = 'Best P value'
    def __init__(self, mode, decomposer_type, rank):
        self.mode = mode
        self._name = f'Best P value for mode {mode}'
        decomposition_type = getattr(pytensor.decomposition,decomposer_type).DecompositionType
        self.decomposition_type = decomposition_type
        #self.decomposer_type = decomposer_type
        self.rank = rank
        #self.decomposition_type = None
        #decomposition = self.decomposer_type.load_from_hdf5_group()

    def _evaluate(self, data_reader, h5):
         
        #decomposer = getattr(pytensor.decomposition, self.decomposer_type)(rank=self.rank,max_its=1000)
        #decomposer.decomposition.load_from_hdf5_group(h5)
        decomposition = self.decomposition_type.load_from_hdf5_group(h5)
        factors = decomposition.factor_matrices[self.mode]

        classes = data_reader.classes

        assert len(set(classes)) == 2

        indices = [[i for i, c in enumerate(classes) if c == class_] for class_ in set(classes)]
        p_values = tuple(ttest_ind(factors[indices[0]], factors[indices[1]], equal_var=False).pvalue)
        return min(p_values)
