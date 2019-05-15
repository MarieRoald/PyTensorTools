from abc import ABC, abstractproperty, abstractmethod
from scipy.io import loadmat
import numpy as np

# en liste med classes for hver mode?
# En liste med dictionaries med labelinfo som f.eks. site for hver mode?
# - kanskje vi kan putte on off som labels her også?



class BaseDataReader(ABC):
    @abstractmethod
    def __init__(self, mode_names=None):
        self._tensor = None
        self._classes = None
        self.mode_names = mode_names

    @property
    def tensor(self):
        return self._tensor
    
    @property
    def classes(self):
        return self._classes
        


class MatlabDataReader(BaseDataReader):
    def __init__(self, file_path, tensor_name, classes=None, mode_names=None):
        """Example:
            dataset = MatlabDataReader('./data.mat', 'data', [{}, {'schizophrenia': 'classes'}, {}], ['voxel', 'patient', 'time'])
        """
        super().__init__(mode_names=mode_names)
        self.file_path = file_path
        self._tensor = np.array(loadmat(file_path, variable_names=[tensor_name])[tensor_name])
        
        if classes is not None:
            self._classes = [{} for _ in self._tensor.shape]
            for class_dict, mode_classes in zip(self._classes, classes):
                for name, varname in mode_classes.items():
                    class_dict[name] = np.array(loadmat(file_path, variable_names=[varname])[varname])
