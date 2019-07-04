#!/home/marie/anaconda3/bin/python
import sys
import argparse
from pathlib import Path
import json


sys.path.append('../PyTensor')
sys.path.append('../PyTensor_classification')
sys.path.append('../PlotTools')

import pytensor.base
from pytensortools.experiment import Experiment
from pytensortools.evaluation.experiment_evaluator import ExperimentEvaluator
from pytensortools import summary_writers

sys.path.append('../PlotTools')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_path', type=str)
    parser.add_argument('evaluator_params', type=str)
    parser.add_argument('--is_single', type=bool, help='Whether subfolders should be iterated over.', default=False)
    args = parser.parse_args()

    with open(args.evaluator_params) as f:
        evaluator_params = json.load(f)

    evaluator = ExperimentEvaluator(
        **evaluator_params
    )
    # '/home/mariero/experiment_logs/MCIC/CP_ALS/CP_ALS_rank_2_01'
    
    experiments = sorted(
        filter(
            lambda x: x.is_dir(), Path(args.result_path).iterdir()
        )
    )
    if args.is_single:
        experiments = [Path(args.result_path)]

    for experiment in experiments:
        print(f'Evaluating {experiment}')
        if not (experiment/'summaries'/'summary.json').is_file():
            print(f'Skipping {experiment}')
            continue
        evaluator.evaluate_experiment(str(experiment))
        summary_writers.create_spreadsheet(Path(experiment))

    if not args.is_single:
        summary_writers.create_csv(Path(args.result_path), new_file=True)
        summary_writers.create_ppt(Path(args.result_path))
