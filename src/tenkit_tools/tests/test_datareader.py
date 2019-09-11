import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ..datareader import MatlabDataReader


class TestMatlabDataReader:

    num_samples = 30

    @pytest.fixture
    def random_tensor(self):
        return np.random.randn(self.num_samples,20,40)

    @pytest.fixture
    def random_classes(self):
        return np.random.randint(0, high=2, size=self.num_samples)


    def test_stored_tensor_loads_correctly(self, random_tensor):
        with tempfile.TemporaryDirectory() as tmpdirname:
            matlab_file_path = Path(tmpdirname)/Path('tmp_tensor.mat')

            savemat(matlab_file_path, {'tensor': random_tensor})

            dr = MatlabDataReader(matlab_file_path, 'tensor')
            np.allclose(random_tensor, dr.tensor)

    def test_stored_classes_loads_correctly(self, random_tensor, random_classes):
        with tempfile.TemporaryDirectory() as tmpdirname:
            matlab_file_path = Path(tmpdirname)/Path('tmp_tensor_with_class.mat')

            savemat(matlab_file_path, {'tensor': random_tensor, 'classes': random_classes})

            dr = MatlabDataReader(matlab_file_path, 'tensor', 'classes')
        
            np.allclose(random_tensor, dr.tensor)
            np.allclose(random_classes, dr.classes)
