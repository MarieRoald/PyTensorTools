from abc import ABC, abstractmethod
from typing import Dict
from subclass_register import SubclassRegister
import warnings

import numpy as np

from . import datareader
from .utils import TestDefaults

test_defaults = TestDefaults()
preprocessor_register = SubclassRegister('preprocessor')


def generate_data_reader(data_reader_params, preprocessor_params):
    warnings.warn('This should be somewhere else or renamed', RuntimeWarning)
    DataReader = getattr(datareader, data_reader_params["type"])
    data_reader = DataReader(**data_reader_params["arguments"])
    data_reader = preprocess_data(data_reader, preprocessor_params)

    return data_reader

def preprocess_data(data_reader, preprocessor_params):
    if preprocessor_params is not None:
        if isinstance(preprocessor_params, Dict):
            preprocessor_params = [preprocessor_params]

        for preprocessor_params in preprocessor_params:
            Preprocessor = globals()[preprocessor_params["type"]]
            args = preprocessor_params.get("arguments", {})
            data_reader = Preprocessor(data_reader=data_reader, **args)
    return data_reader


def get_preprocessor(preprocessor):
    raise NotImplementedError


@preprocessor_register.link_base
class BasePreprocessor(datareader.BaseDataReader):
    def __init__(self, data_reader):
        self.data_reader = data_reader
        self.mode_names = data_reader.mode_names
        tensor, classes = self.preprocess(data_reader)
        self._tensor, self._classes = tensor, classes

    @abstractmethod
    def preprocess(self, data_reader):
        return data_reader.tensor, data_reader.classes

    @property
    def tensor(self):
        return self._tensor

    @property
    def classes(self):
        return self._classes


@test_defaults.set_default({})
class IdentityMap(BasePreprocessor):
    def preprocess(self, data_reader):
        return super().preprocess(data_reader)


@test_defaults.set_default({'center_across':0})
class Center(BasePreprocessor):
    """Center data across given mode. 
    
    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    center_across: int
        Which mode to center across
    """
    def __init__(self, data_reader, center_across):
        self.center_across = center_across
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        tensor = tensor - tensor.mean(axis=self.center_across, keepdims=True)
        return tensor, data_reader.classes


@test_defaults.set_default({})
class GlobalScale(BasePreprocessor):
    """Globally scale the tensor to have unit frobenius norm.

    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    """
    def __init__(self, data_reader):
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        norm = np.linalg.norm(tensor.ravel())  # Failed with fro for more than two modes.
        tensor = tensor / norm

        return tensor, data_reader.classes


@test_defaults.set_default({'scale_within':1})
class Scale(BasePreprocessor):
    """Scale data within given mode. 
    
    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    scale_within: int
        Which mode to scale within
    """
    def __init__(self, data_reader, scale_within):
        self.scale_within = scale_within
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        reduction_axis = [i for i in range(len(tensor.shape)) if i != self.scale_within]
        weightings = np.linalg.norm(tensor, axis=tuple(reduction_axis), keepdims=True)
        tensor = tensor / weightings

        return tensor, data_reader.classes


@test_defaults.set_default({'center_across':0, 'scale_within':1})
class Standardize(BasePreprocessor):
    def __init__(self, data_reader, center_across, scale_within):
        if center_across == scale_within:
            raise ValueError(
                "Cannot scale across the same mode as we center within.\n"
                "See Centering and scaling in component analysis by R Bro and AK Smilde, 1999"
            )
        self.center_across = center_across
        self.scale_within = scale_within
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        centered_dataset = Center(data_reader, self.center_across)
        scaled_dataset = Scale(centered_dataset, self.scale_within)
        return scaled_dataset.tensor, scaled_dataset.classes


@test_defaults.set_default({'mode':1})
class MarylandPreprocess(BasePreprocessor):
    
    def __init__(self, data_reader, mode, center=True, scale=True):
        self.mode = mode
        self.center = center
        self.scale = scale
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        if self.center:
            tensor = data_reader.tensor - data_reader.tensor.mean(
                self.mode, keepdims=True
            )
        if self.scale:
            tensor = tensor / np.linalg.norm(tensor, axis=self.mode, keepdims=True)
        return tensor, data_reader.classes


@preprocessor_register.skip
class BaseRemoveOutliers(BasePreprocessor):
    def __init__(self, data_reader, mode, remove_from_classes=True):
        self.mode = mode
        self.remove_from_classes = remove_from_classes
        super().__init__(data_reader)

    def _delete_idx(self, data_reader, delete_idx):
        tensor = data_reader.tensor
        classes = data_reader.classes

        processed_tensor = np.delete(tensor, delete_idx, axis=self.mode)

        if data_reader.classes is not None and self.remove_from_classes:
            processed_classes = [classes for classes in data_reader.classes]
            processed_classes[self.mode] = {
                name: np.delete(value, delete_idx)
                for name, value in processed_classes[self.mode].items()
            }
        else:
            processed_classes = classes

        return processed_tensor, processed_classes


@test_defaults.set_default({'mode':1, 'outlier_idx':[1,4,5]})
class RemoveOutliers(BaseRemoveOutliers):
    """Removes subarrays at the given indices across a given mode. 

    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    mode: int
        Which mode to remove subarrays from
    outlier_idx: np.ndarray
        Indices to remove
    remove_from_classes: bool (optinal)
        By default, the class value is removed from the class array. 
        if false it is not removed (usefull if the class value is not in the array)
    """
    def __init__(self, data_reader, mode, outlier_idx, remove_from_classes=True):
        self.outlier_idx = outlier_idx
        super().__init__(data_reader, mode, remove_from_classes=remove_from_classes)

    def preprocess(self, data_reader):
        return self._delete_idx(data_reader, self.outlier_idx)


@test_defaults.set_default({'mode':1, 'start_idx':1, 'end_idx':-2})
class RemoveRangeOfOutliers(BaseRemoveOutliers):
    """Removes subarrays that lies within a range across a given mode.

    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    mode: int
        Which mode to remove subarrays from
    start_idx: int
        First element to delete
    end_idx: int
        Endpoint of deletion. This element will not be deleted
    remove_from_classes: bool (optinal)
        By default, the class value is removed from the class array. 
        if false it is not removed (useful if the class value is not in the array)
    """
    def __init__(self, data_reader, mode, start_idx, end_idx, remove_from_classes=True):
        self.start_idx = start_idx
        self.end_idx = end_idx
        super().__init__(data_reader, mode, remove_from_classes=remove_from_classes)

    def preprocess(self, data_reader):
        delete_idx = range(self.start_idx, self.end_idx)
        return self._delete_idx(data_reader, delete_idx)


@test_defaults.set_default({'mode':1, 'class_name':'class', 'class_to_remove':0})
class RemoveClass(BaseRemoveOutliers):
    """Removes subarrays that matches a given class across a given mode.

    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    mode: int
        Which mode to remove subarrays from
    class_name: str
        Name of the class to match against
    class_to_remove: str
        Which class value should be removed
    remove_from_classes: bool (optinal)
        By default, the class value is removed from the class array. 
        if false it is not removed (usefull if the class value is not in the array)
    """
    def __init__(
        self, data_reader, mode, class_name, class_to_remove, remove_from_classes=True
    ):
        self.class_name = class_name
        self.class_to_remove = class_to_remove
        super().__init__(data_reader, mode, remove_from_classes=remove_from_classes)

    def preprocess(self, data_reader):
        delete_idx = np.where(
            data_reader.classes[self.mode][self.class_name] == self.class_to_remove
        )
        return self._delete_idx(data_reader, delete_idx)


@test_defaults.set_default({})
class Transpose(BasePreprocessor):
    """Permutes the modes of a tensor.

    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    permutation : list of ints (optional)
        Permutes the modes according to the values given.
        If not given,reverse the dimensions, otherwise permute the axes according to the values given.
    """
    def __init__(self, data_reader, permutation=None):
        if permutation is None:
            permutation = list(reversed(range(data_reader.tensor.ndim)))
        
        self.permutation = permutation
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        if self.mode_names is not None:
            self.mode_names = [self.mode_names[idx] for idx in self.permutation]
        classes = [data_reader.classes[idx] for idx in self.permutation]
        return np.transpose(data_reader.tensor, self.permutation), classes

@test_defaults.set_default({'noise_level': 0.33})
class AddNoise(BasePreprocessor):
    def __init__(self, data_reader, noise_level):
        self.noise_level = noise_level
        super().__init__(data_reader)
    
    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        noise = np.random.standard_normal(size=data_reader.tensor.shape)

        return tensor + self.noise_level*noise*(np.linalg.norm(tensor)/np.linalg.norm(noise)), data_reader.classes

@test_defaults.set_default({'mode':0})
class Derivative(BasePreprocessor):
    """Takes the derivative across one mode of a tensor.

    Attributes
    ----------
    data_reader : DataReader
        DataReader object containing the dataset
    mode: int
        Which mode to take the derivative across
    """

    def __init__(self, data_reader, mode=0):
        self.mode = mode
        super().__init__(data_reader)

    def tensor_derivative(self, tensor, mode=0):

        num_elements = tensor.shape[mode]
        all_indices = np.arange(num_elements)

        even_indices = all_indices[1:]
        odd_indices = all_indices[:-1]

        derivative_tensor = tensor.take(indices=even_indices, axis=mode) -\
                            tensor.take(indices=odd_indices, axis=mode)

        return derivative_tensor

    def preprocess(self, data_reader):
        derivative_tensor = self.tensor_derivative(data_reader.tensor)

        derivative_classes = data_reader.classes
        derivative_classes[self.mode] = {class_name: self.tensor_derivative(class_values, mode=0) 
                                         for (class_name, class_values) in derivative_classes[self.mode].items()}

        return derivative_tensor, derivative_classes