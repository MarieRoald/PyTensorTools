from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from tenkit_tools.preprocessor import test_defaults
from tenkit_tools.datareader import NumpyDataReader

#for cls, kwargs in test_defaults.items():
#    instans = cls(**kwargs)
#    print(instans)

@pytest.fixture
def num_samples():
    return 30

@pytest.fixture
def random_tensor(num_samples):
    return np.random.randn(num_samples, 20, 40)

@pytest.fixture
def random_classes(num_samples):
    return [{}, {'class':np.random.randint(0, high=2, size=num_samples)}, {}]

@pytest.fixture
def random_datareader(random_tensor, random_classes):
    return NumpyDataReader(tensor=random_tensor, classes=random_classes)

def test_preprocessor_creation(random_datareader):
    for Preprocessor, kwargs in test_defaults.items():
        preprocessor = Preprocessor(random_datareader, **kwargs)
