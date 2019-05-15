from abc import ABC, abstractmethod
import numpy as np


def get_preprocessor(preprocessor):
    raise NotImplementedError


class BasePreprocessor(ABC):
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
                'Cannot scale across the same mode as we center within.\n'
                'See Centering and scaling in component analysis by R Bro and AK Smilde, 1999'
            )
        self.center_across = center_across
        self.scale_within = scale_within
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        centered_dataset = Center(data_reader, self.center_across)
        scaled_dataset = Scale(centered_dataset, self.scale_within)
        return scaled_dataset.tensor, scaled_dataset.classes


class MarylandPreprocess(BasePreprocessor):
    def __init__(self, data_reader, mode):
        self.mode = mode
        super().__init__(data_reader)
    
    def preprocess(self, data_reader):
        centered = data_reader.tensor - data_reader.tensor.mean(self.mode, keepdims=True)
        scaled = centered/np.linalg.norm(centered, axis=self.mode, keepdims=True)
        return scaled, data_reader.classes

class RemoveOutliers(BasePreprocessor):
    # TODO: this only works if classes match the mode we remove outliers from!
    def __init__(self, data_reader, mode, outlier_idx, remove_from_classes=True):
        self.outlier_idx = outlier_idx
        self.mode = mode
        self.remove_from_classes = remove_from_classes
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        classes = data_reader.classes

        processed_tensor = np.delete(tensor, self.outlier_idx, axis=self.mode)
        if data_reader.classes is not None and self.remove_from_classes:
            processed_classes = [classes for classes in data_reader.classes]
            processed_classes[self.mode] = {
                name: np.delete(value, self.outlier_idx) 
                    for name, value in processed_classes[self.mode].items()
            }
        else:
            processed_classes = classes
        
        return processed_tensor, processed_classes

class RemoveRangeOfOutliers(BasePreprocessor):
    def __init__(self, data_reader, mode, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.mode = mode
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        tensor = data_reader.tensor
        classes = data_reader.classes

        processed_tensor = np.delete(tensor, range(self.start_idx, self.end_idx), axis=self.mode)
        if data_reader.classes is not None:
            processed_classes = [classes for classes in data_reader.classes]
            processed_classes[self.mode] = {
                name: np.delete(value, range(self.start_idx, self.end_idx)) 
                    for name, value in processed_classes[self.mode].items()
            }
        else:
            processed_classes = classes
        
        return processed_tensor, processed_classes


class Transpose(BasePreprocessor):
    def __init__(self, data_reader, permutation):
        self.permutation = permutation
        super().__init__(data_reader)

    def preprocess(self, data_reader):
        self.mode_names = [self.mode_names[idx] for idx in self.permutation]
        classes = [data_reader.classes[idx] for idx in self.permutation]
        return np.transpose(data_reader.tensor, self.permutation), classes

