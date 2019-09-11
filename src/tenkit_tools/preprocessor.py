from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from .datareader import BaseDataReader


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


class BasePreprocessor(BaseDataReader):
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


class IdentityMap(BasePreprocessor):
    def preprocess(self, data_reader):
        return super().preprocess(data_reader)


class Center(BasePreprocessor):
    def __init__(self, data_reader, center_across):
        self.center_across = center_across
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        tensor = tensor - tensor.mean(axis=self.center_across, keepdims=True)
        return tensor, data_reader.classes


class Scale(BasePreprocessor):
    def __init__(self, data_reader, scale_within):
        self.scale_within = scale_within
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        reduction_axis = [i for i in range(len(tensor.shape)) if i != self.scale_within]
        weightings = np.linalg.norm(tensor, axis=reduction_axis, keepdims=True)
        tensor = tensor / weightings

        return tensor, data_reader.classes


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


class RemoveOutliers(BaseRemoveOutliers):
    def __init__(self, data_reader, mode, outlier_idx, remove_from_classes=True):
        self.outlier_idx = outlier_idx
        super().__init__(data_reader, mode, remove_from_classes=remove_from_classes)

    def preprocess(self, data_reader):
        return self._delete_idx(data_reader, self.outlier_idx)


class RemoveRangeOfOutliers(BaseRemoveOutliers):
    def __init__(self, data_reader, mode, start_idx, end_idx, remove_from_classes=True):
        self.start_idx = start_idx
        self.end_idx = end_idx
        super().__init__(data_reader, mode, remove_from_classes=remove_from_classes)

    def preprocess(self, data_reader):
        delete_idx = range(self.start_idx, self.end_idx)
        return self._delete_idx(data_reader, delete_idx)


class RemoveClass(BaseRemoveOutliers):
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


class Transpose(BasePreprocessor):
    def __init__(self, data_reader, permutation):
        self.permutation = permutation
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        self.mode_names = [self.mode_names[idx] for idx in self.permutation]
        classes = [data_reader.classes[idx] for idx in self.permutation]
        return np.transpose(data_reader.tensor, self.permutation), classes
