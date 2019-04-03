import sys
sys.path.append('/home/marie/Dropbox/Programming/Simula/PyTensor_classification/')
import pytensor.base
from pytensortools.experiment import Experiment
from pytensortools.evaluation.experiment_evaluator import ExperimentEvaluator


if __name__ == '__main__':
    single_run_evaluators = [
        {'type': 'FinalLoss', 'arguments': {}},
        {'type': 'ExplainedVariance', 'arguments': {}},
        {'type': 'PValue', 
         'arguments': {
             'mode': 0,
             'decomposer_type': 'CP_ALS',
             'rank': 4
         }
        }
    ]

    evaluator = ExperimentEvaluator(single_run_evaluators)
    evaluator.evaluate_experiment('test_result_for_evaluation_test')