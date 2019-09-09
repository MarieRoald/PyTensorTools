from abc import ABC, abstractmethod
from .. import evaluation
import tenkit
from .. import postprocessor


def create_evaluator(evaluator_params, summary, postprocessor_params, data_reader,):
    Evaluator = getattr(evaluation, evaluator_params['type'])
    kwargs = evaluator_params.get('arguments', {})
    return Evaluator(summary=summary, postprocessor_params=postprocessor_params, data_reader=data_reader, **kwargs)


def create_evaluators(evaluators_params, summary, postprocessor_params, data_reader,):
    evaluators = []
    for evaluator_params in evaluators_params:
        evaluators.append(create_evaluator(evaluator_params, summary, postprocessor_params, data_reader,))
    return evaluators


class BaseEvaluator(ABC):
    _name = None
    def __init__(self, summary, post_processor_params=None, datareader=None):
        self.summary = summary
        self.DecomposerType = getattr(tenkit.decomposition, summary['model_type'])
        self.DecompositionType = self.DecomposerType.DecompositionType
        self.post_processor_params = post_processor_params
        if self.post_processor_params is None:
            self.post_processor_params = []
        self.datareader = datareader

    @property
    def name(self):
        if self._name is None:
            return type(self).__name__
        return self._name
    
    def load_final_checkpoint(self, h5):
        final_it = h5.attrs['final_iteration']
        postprocessed =  self.DecompositionType.load_from_hdf5_group(h5[f'checkpoint_{final_it:05d}'])

        for postprocessor_params in self.post_processor_params:
            PostprocessorType = getattr(postprocessor, postprocessor_params['type'])
            kwargs = postprocessor_params.get('arguments', {})
            postprocessed = PostprocessorType(postprocessed, self.datareader, **kwargs)
        
        return postprocessed