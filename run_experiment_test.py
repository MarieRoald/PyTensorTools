from scipy.io import savemat
sys.path.append('/home/marie/Dropbox/Programming/Simula/PyTensor_classification/')
import pytensor.base
from pytensortools.experiment import Experiment
import numpy as np

if __name__ == "__main__":

    data_reader_params = {
        'type': 'MatlabDataReader',
        'arguments': {
            'file_path': 'x.mat',
            'tensor_name': 'X',
            'classes_name': 'classes'
        }
    }

    logger_params = [
        {
            'type': 'LossLogger',
        },
        {
            'type': 'ExplainedVarianceLogger',
        }
    ]

    experiment_params = {
        'num_runs': 10,
        'save_path': 'test_result_for_evaluation_test'
    }

    decomposer_params = {
        'type': 'CP_ALS',
        'arguments': {
            'rank': 4,
            'max_its': 10,
            'checkpoint_frequency': 2
        },
        'fit_params': {

        }

    }
    ktensor = pytensor.base.KruskalTensor.random_init((100, 20, 300), rank=4)
    ktensor.store('ktensor.h5')
    X = ktensor.construct_tensor()
    c = np.random.randint(0,2, size=(100,))
    print(c)
    savemat('x.mat', {'X': X, 'classes':c})
    print('Starting dataset')
    experiment = Experiment(experiment_params, data_reader_params, decomposer_params, logger_params)
    runs = experiment.run_experiments()
    for run in runs:
        print(run.loss())
    


    #parametere jeg vil gi:
    # - Rank
    # - Vilken modell
    # - Hvor mange runs
    # - Hva er toleransen?