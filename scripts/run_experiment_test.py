import sys

import numpy as np
from scipy.io import savemat

import tenkit.base
from tenkittools.experiment import Experiment

sys.path.append('/home/marie/Dropbox/Programming/Simula/PyTensor_classification/')
sys.path.append('../PyTensor/')

if __name__ == "__main__":

    data_reader_params = {
        'type': 'MatlabDataReader',
        'arguments': {
            'file_path': 'x.mat',
            'tensor_name': 'X',
            'classes': [{'test': 'classes'}, {'tast': 'labels'}, {}],
            'mode_names': ['test', 'tast', 'kronk']
        }
    }

    preprocessor_params = {
        'type': 'Center',
        'arguments': {
            'center_across': 0
        }
    }

    logger_params = [
        {
            'type': 'LossLogger',
        },
        {
            'type': 'ExplainedVarianceLogger',
        },
    ]

    experiment_params = {
        'num_runs': 10,
        'save_path': 'logs',
        'experiment_name': 'test_run'
    }

    decomposer_params = {
        'type': 'CP_ALS',
        'arguments': {
            'rank': 4,
            'max_its': 11,
            'checkpoint_frequency': 2
        },
        'fit_params': {

        }

    }
    ktensor = tenkit.base.KruskalTensor.random_init((100, 20, 300), rank=4)
    ktensor.store('ktensor.h5')
    X = ktensor.construct_tensor()
    c = np.random.randint(0, 2, size=(100,))
    labels = ['hei', 'hoi', 'halla', 'heissann']
    l = sorted([labels[np.random.randint(0, 4)] for _ in range(X.shape[1])])
    print(c)
    savemat('x.mat', {'X': X, 'classes':c, 'labels': l})
    print('Starting dataset')
    experiment = Experiment(experiment_params, data_reader_params, decomposer_params, logger_params, preprocessor_params=preprocessor_params)
    # runs = experiment.run_experiments()
    experiment.data_reader.to_matlab([[], ['tast'], []], 'test.mat')
    


    #parametere jeg vil gi:
    # - Rank
    # - Vilken modell
    # - Hvor mange runs
    # - Hva er toleransen?
