#!/home/marie/anaconda3/bin/python
import sys
sys.path.append('../PyTensor')
sys.path.append('../PyTensor_classification')
import pytensor.base
from pytensortools.experiment import Experiment
from pytensortools.evaluation.experiment_evaluator import ExperimentEvaluator
import argparse


if __name__ == '__main__':
    single_run_evaluators = [
        {'type': 'FinalLoss', 'arguments': {}},
        {'type': 'ExplainedVariance', 'arguments': {}},
        {'type': 'PValue', 
         'arguments': {
            'mode': 2,
            'class_name': 'schizophrenic'
         }
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
                'class_name': 'schizophrenic'
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
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    evaluator = ExperimentEvaluator(
        single_run_evaluator_params=single_run_evaluators,
        multi_run_evaluator_params=multi_run_evaluators,
        single_run_visualiser_params=single_run_visualisers,
    )
    # '/home/mariero/experiment_logs/MCIC/CP_ALS/CP_ALS_rank_2_01'
    evaluator.evaluate_experiment(args.path)
