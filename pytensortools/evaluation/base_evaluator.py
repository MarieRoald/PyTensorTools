from abc import ABC, abstractmethod
from .. import evaluation
import pytensor


def create_evaluator(evaluator_params, summary):
    Evaluator = getattr(evaluation, evaluator_params['type'])
    kwargs = evaluator_params.get('arguments', {})
    return Evaluator(summary=summary, **kwargs)


def create_evaluators(evaluators_params, summary):
    evaluators = []
    for evaluator_params in evaluators_params:
        evaluators.append(create_evaluator(evaluator_params, summary))
    return evaluators


class BaseEvaluator(ABC):
    _name = None
    def __init__(self, summary):
        self.summary = summary
        self.DecomposerType = getattr(pytensor.decomposition, summary['model_type'])
        self.DecompositionType = self.DecomposerType.DecompositionType

    @property
    def name(self):
        if self._name is None:
            return type(self).__name__
        return self._name
    
    def load_final_checkpoint(self, h5):
        final_it = h5.attrs['final_iteration']
        return self.DecompositionType.load_from_hdf5_group(h5[f'checkpoint_{final_it:05d}'])