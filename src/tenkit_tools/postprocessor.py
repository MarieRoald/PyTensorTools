from abc import ABC, abstractmethod
from tenkit.decomposition import decompositions
from . import utils



class BasePostprocessor(ABC):
    Decomposition = decompositions.BaseDecomposedTensor
    def __init__(self, decomposition, data_reader):
        self.data_reader = data_reader
        self.rank = decomposition.rank
        self.decomposition = self.postprocess(decomposition)

    @abstractmethod
    def postprocess(self, decomposition):
        raise NotImplementedError

    def __getitem__(self, value):
        return self.Decomposition.__getitem__(self, value)
    
    @property
    def shape(self):
        return self.Decomposition.shape(self)
    
    def construct_tensor(self):
        return self.Decomposition.construct_tensor(self)
    

class KruskalPostprocessor(BasePostprocessor):
    Decomposition = decompositions.KruskalTensor

    def __init__(self, decomposition, data_reader):
        self.factor_matrices = decomposition.factor_matrices
        self.weights = decomposition.weights
        super().__init__(decomposition, data_reader)


class KruskalSignFlipper(KruskalPostprocessor):
    def __init__(self, decomposition, data_reader, flip_params: dict, correction_mode: int):
        """
        flip specification dictionaries:
        {
            'method': 'classification_driven',
            'arguments': {
                'class_name': 'schizophrenic',
                'positive_label_value': '2'
            }
        }
        {
            'method': 'weight_drive'
        }
        {
            'method':
        }

        flip_params
            Dictionary, mapping mode idx to flip specification dictionary

        correction_mode
            Index of mode that should be used to correct the sign of
            the other modes. Cannot be key in flip_params.
        """
        self.flip_params = flip_params
        self.correction_mode = correction_mode
        if correction_mode in self.flip_params:
            raise ValueError(f'Correction mode ({correction_mode}) cannot be amongst the keys of flip params')
        super().__init__(decomposition, data_reader)

    def get_sign(
        self,
        mode,
        method,
        class_name=None,
        positive_label_value=None,
    ):
        factor_matrix = self.factor_matrices[mode]
        if method == 'classification_driven':
            if class_name is None:
                raise ValueError('Must supply values for classification driven sign flip')

            

            labels = self.data_reader.classes[mode][class_name]
            return utils.classification_driven_get_sign(
                factor_matrix, labels, positive_label_value, factor_matrix
            )
        elif method == 'data_driven':
            return utils.data_driven_get_sign(factor_matrix, self.data_reader.tensor)
        elif method == 'sign_driven':
            return utils.sign_driven_get_sign(factor_matrix)
        else:
            raise ValueError(f'{method} is not a valid sign flip method') 
    
    def postprocess(self, decomposition):
        correction_sign = 1
        for flip_mode, params in self.flip_params.items():
            flip_mode = int(flip_mode)
            sign = self.get_sign(flip_mode, params['method'], **params.get('arguments', {}))
            self.factor_matrices[flip_mode] *= sign
            correction_sign *= sign
        
        self.factor_matrices[self.correction_mode] *= sign
