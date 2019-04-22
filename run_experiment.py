import sys
import json
import argparse
from pathlib import Path

sys.path.append('/home/marie/Dropbox/Programming/Simula/PyTensor_classification/')
from pytensortools.experiment import Experiment


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
        default='.'
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
    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)

    with (experiment_path/'data_reader_params.json').open() as f:
        data_reader_params = json.load(f)
    with (experiment_path/'log_params.json').open() as f:
        logger_params = json.load(f)
    with (experiment_path/'decomposition_params.json').open() as f:
        decomposition_params = json.load(f)
    
    if args.rank is not None:
        decomposition_params['arguments']['rank'] = args.rank
    
    if args.max_its is not None:
        decomposition_params['arguments']['max_its'] = args.max_its

    if args.tol is not None:
        decomposition_params['arguments']['tol'] = args.tol

    decomposition_type = decomposition_params['type']
    rank = decomposition_params['arguments']['rank']
    experiment_name = f'{decomposition_type}_rank_{rank}'
    experiment_path = Path(args.save_path)/experiment_name
    experiment_params = {
        'num_runs':args.num_runs,
        'save_path': str(experiment_path)
    }

    experiment = Experiment(experiment_params, data_reader_params, decomposition_params, logger_params)
    runs = experiment.run_experiments()