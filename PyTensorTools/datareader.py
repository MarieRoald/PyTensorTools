from abc import ABC, abstractproperty, abstractmethod
from scipy.io import loadmat 


class BaseDataReader(ABC):
    @abstractmethod
    def __init__(self):
        self._tensor = None
        self._classes = None

    @property
    def tensor(self):
        return self._tensor
    
    @property
    def classes(self):
        return self._classes


class MatlabDataReader(BaseDataReader):
    def __init__(self, matlab_file_path, tensor_name, classes_name=None):
        super().__init__()
        self.file_path = matlab_file_path
        self._tensor = loadmat(matlab_file_path, variable_names=[tensor_name])

        if classes_name is not None:
            self._classes = loadmat(matlab_file_path, variable_names=[classes_name])
        