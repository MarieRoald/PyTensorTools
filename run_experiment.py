import sys
from scipy.io import savemat
sys.path.append('/home/marie/Dropbox/Programming/Simula/PyTensor_classification/')
import pytensor.base
from pytensortools.experiment import Experiment




if __name__ == "__main__":

    data_reader_params = {
        'type': 'MatlabDataReader',
        'arguments': {
            'file_path': 'x.mat',
            'tensor_name': 'X'
        }
    }

    logger_params = [
        {
            'type': 'LossLogger',
        }
    ]

    experiment_params = {
        'num_runs': 10,
        'save_path': 'test_result'
    }

    decomposer_params = {
        'type': 'CP_ALS',
        'arguments': {
            'rank': 4,
            'max_its': 1000,
            'checkpoint_period': 2
        },
        'fit_params': {

        }

    }
    ktensor = pytensor.base.KruskalTensor.random_init((100, 2000, 300), rank=4)
    ktensor.store('ktensor.h5')
    X = ktensor.construct_tensor()
    savemat('x.mat', {'X': X})
    print('Starting dataset')
    experiment = Experiment(experiment_params, data_reader_params, decomposer_params, logger_params)
    runs = experiment.run_experiments()
    for run in runs:
        print(run.MSE)

    #parametere jeg vil gi:
    # - Rank
    # - Vilken modell
    # - Hvor mange runs
    # - Hva er toleransen?