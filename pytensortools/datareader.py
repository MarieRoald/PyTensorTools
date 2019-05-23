from abc import ABC, abstractproperty, abstractmethod
from scipy.io import loadmat
import numpy as np
from scipy.stats import ttest_ind
import h5py

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


class HDF5DataReader(BaseDataReader):
    def _load_h5_dataset(self, file_path, dataset):
        with h5py.File(file_path, 'r') as h5:
            return h5[dataset][...]

    def _load_data_tensor(self, file_path, tensor_name):
        return self._load_h5_dataset(file_path, tensor_name)

    def _load_class(self, file_path, class_name):
        return self._load_h5_dataset(file_path, class_name)

    def _load_meta_data(self, file_path, classes):
        _classes = [{} for _ in self._tensor.shape]
        for class_dict, mode_classes in zip(_classes, classes):
            for name, varname in mode_classes.items():
                class_dict[name] = self._load_class(file_path, varname)
        return _classes    
        
    def __init__(self, file_path, tensor_name, meta_info_path=None, classes=None, mode_names=None):
        super().__init__(mode_names=mode_names)
        self.file_path = file_path

        if meta_info_path is None:
            meta_info_path = file_path

        self.meta_info_path = meta_info_path
        self._tensor = self._load_data_tensor(self.file_path, tensor_name)

        if classes is not None:
            self._classes = self._load_meta_data(self.meta_info_path, classes)
