import sys
import json
import argparse
from pathlib import Path

sys.path.append('/home/marie/Dropbox/Programming/Simula/PyTensor_classification/')
sys.path.append('../PyTensor/')
from pytensortools.experiment import Experiment

#TODO: Add argument to assert that decomposer params is set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment_path',
        help='Path to a folder containing the following files:\n'
             '\tdata_reader_params.json\n'
             '\tlog_params.json\n'
             '\tdecomposition_params.json\n',
        type=str
    )
    parser.add_argument('-n', '--num_runs', 
        help="The number of runs to perform, default=60", 
        type=int,
        default=60
    )
    parser.add_argument('-s', '--save_path', 
        help="The location to store the experiment results, default='.'", 
        type=str,
        default=None
    )
    parser.add_argument(
        '-r',
        '--rank',
        help='The rank of the decomposition',
        type=int,
        default=None
    )
    parser.add_argument(
        '--tol',
        help='The convergence tolerance',
        type=float,
        default=None
    )
    parser.add_argument(
        '--max_its',
        help='The maximum number of iterations to run.',
        default=None,
        type=int
    )
    parser.add_argument(
        '--suffix',
        help='The suffix of the experiment name',
        default="",
        type=str
    )

    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)

    with (experiment_path/'data_reader_params.json').open() as f:
        data_reader_params = json.load(f)
    with (experiment_path/'log_params.json').open() as f:
        logger_params = json.load(f)
    with (experiment_path/'decomposition_params.json').open() as f:
        decomposition_params = json.load(f)
    preprocessor_params = None
    if (experiment_path/'preprocessor_params.json').is_file():
        with (experiment_path/'preprocessor_params.json').open() as f:
            preprocessor_params = json.load(f)

    if args.save_path is not None:
        save_path = args.save_path
    else:
        with (experiment_path/'save_path.json').open() as f:
            save_path = json.load(f)
            if 'save_path' in save_path:
                save_path = save_path['save_path']
            else:
                save_path = '.'
    
    if args.rank is not None:
        decomposition_params['arguments']['rank'] = args.rank
    
    if args.max_its is not None:
        decomposition_params['arguments']['max_its'] = args.max_its

    if args.tol is not None:
        decomposition_params['arguments']['convergence_tol'] = args.tol

    decomposition_type = decomposition_params['type']
    rank = decomposition_params['arguments']['rank']
    experiment_params = {
        'num_runs': args.num_runs,
        'save_path': str(save_path),
        'experiment_name': experiment_path.name
    }

    experiment = Experiment(
        experiment_params,
        data_reader_params,
        decomposition_params,
        logger_params,
        preprocessor_params=preprocessor_params
    )
    runs = experiment.run_experiments()
