#!/home/marie/anaconda3/bin/python
import sys
sys.path.append('../PyTensor')
sys.path.append('../PyTensor_classification')
import pytensor.base
from pytensortools.experiment import Experiment
from pytensortools.evaluation.experiment_evaluator import ExperimentEvaluator
from pytensortools import summary_writers
import argparse
from pathlib import Path


if __name__ == '__main__':
    single_run_evaluators = [
        {'type': 'FinalLoss', 'arguments': {}},
        {'type': 'ExplainedVariance', 'arguments': {}},
        {'type': 'MinPValue', 
         'arguments': {
            'mode': 2,
            'class_name': 'schizophrenic'
         }
        },
	{'type': 'AllPValues',
	 'arguments': {'mode': 2, 'class_name': 'schizophrenic'}
	},
        {'type': 'WorstDegeneracy',
          'arguments': {}
        },
        {'type': 'CoreConsistency',
          'arguments': {}
        },
        {'type': 'MaxKMeansAcc', 
         'arguments': {
             'matlab_scripts_path': 'pytensortools/evaluation/legacy_matlab_code',
             'mode': 2,
             'class_name': 'schizophrenic'

         }
        }
    ]
    multi_run_evaluators = [
      {'type': 'Uniqueness', 'arguments': {}}
    ]
    single_run_visualisers = [
        {
            'type': 'FactorLinePlotter', 
            'arguments': {
                'modes': [0, 1, 2]
            }
        },
        {
            'type': 'FactorScatterPlotter', 
            'arguments': {
                'mode': 2,
                'class_name': 'sites',
		'filename': 'sites_scatter'
            }
        },
        {
            'type': 'ClassLinePlotter', 
            'arguments': {
                'mode': 2,
                'class_name': 'sites',
		'filename': 'sites_line'
            }
        },
        {
            'type': 'FactorScatterPlotter', 
            'arguments': {
                'mode': 2,
                'class_name': 'schizophrenic'
            }
        },
	{
	    'type': 'SingleComponentLinePlotter',
            'arguments' : {
                'mode': 0,
                'filename': 'time_mode'
            }
        },
        {
            'type': 'LogPlotter',
            'arguments': {
                'log_name': 'ExplainedVariance',
                'logger_name': 'ExplainedVarianceLogger',
                'filename': 'explained_variance'
            }
        },
        {
            'type': 'LogPlotter',
            'arguments': {
                'log_name': 'Loss',
                'logger_name': 'LossLogger',
                'filename': 'loss'
            }
        }
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('result_path', type=str)
    parser.add_argument('--is_single', type=bool, help='Whether subfolders should be iterated over.', default=False)
    args = parser.parse_args()

    evaluator = ExperimentEvaluator(
        single_run_evaluator_params=single_run_evaluators,
        multi_run_evaluator_params=multi_run_evaluators,
        single_run_visualiser_params=single_run_visualisers,
    )
    # '/home/mariero/experiment_logs/MCIC/CP_ALS/CP_ALS_rank_2_01'
    
    experiments = sorted(
        filter(
            lambda x: x.is_dir(), Path(args.result_path).iterdir()
        )
    )
    for experiment in experiments:
        print(f'Evaluating {experiment}')
        if not (experiment/'summaries'/'summary.json').is_file():
            print(f'Skipping {experiment}')
            continue
        evaluator.evaluate_experiment(str(experiment))
        summary_writers.create_spreadsheet(Path(experiment))

    if not args.is_single:
        summary_writers.create_csv(Path(args.result_path))
        summary_writers.create_ppt(Path(args.result_path))
