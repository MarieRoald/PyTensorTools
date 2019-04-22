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
             '\tdecomposition_params.json\n'
             '\texperiment_params.json\n',
        type=str
    )
    parser.add_argument(
        '-r',
        '--rank',
        help='The rank of the decomposition',
        type=int,
        default=None
    )
    parser.add_argument(
        '-s',
        '--save_path',
        help='The path to store the experiment output',
        type=str,
        default=None
    )
    args = parser.parse_args()
    experiment_path = Path(args.experiment_path)

    with (experiment_path/'data_reader_params.json').open() as f:
        data_reader_params = json.load(f)
    with (experiment_path/'log_params.json').open() as f:
        logger_params = json.load(f)
    with (experiment_path/'decomposition_params.json').open() as f:
        decompostion_params = json.load(f)
    with (experiment_path/'experiment_params.json').open() as f:
        experiment_params = json.load(f)
    
    if args.rank is not None:
        decomposition_params['arguments']['rank'] = args.rank
    
    if args.save_path is not None:
        experiment_params['save_path'] = args.savepath

    experiment = Experiment(experiment_params, data_reader_params, decompostion_params, logger_params)
    runs = experiment.run_experiments()