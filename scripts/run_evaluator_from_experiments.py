
import argparse
#!/home/marie/anaconda3/bin/python
import sys

import tenkit.base
from tenkittools.evaluation.experiment_evaluator import ExperimentEvaluator
from tenkittools.experiment import Experiment

sys.path.append('../PyTensor')
sys.path.append('../PyTensor_classification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', type=str)
    parser.add_argument('result_parent_path', type=str)
    args = parser.parse_args()
    evaluator_path = Path(args.experiment_path)/'evaluation_params'

    with (evaluator_path/'single_run_evaluators.json').open() as f:
        single_run_evaluator = json.load(f)
    with (evaluator_path/'multi_run_evaluators.json').open() as f:
        multi_run_evaluator = json.load(f)
    with (evaluator_path/'single_run_visualisers.json').open() as f:
        single_run_visualiser = json.load(f)

    evaluator = ExperimentEvaluator(
        single_run_evaluator_params=single_run_evaluators,
        multi_run_evaluator_params=multi_run_evaluators,
        single_run_visualiser_params=single_run_visualisers,
    )
    # '/home/mariero/experiment_logs/MCIC/CP_ALS/CP_ALS_rank_2_01'
    
    experiments = sorted(
        filter(
            lambda x: x.is_dir(), args.result_parent_path.iterdir()
        )
    )
    for experiment in experiments:
        evaluator.evaluate_experiment(str(experiment))
        summary_writers.create_spreadsheet(experiment_path)

    summary_writers.create_csv(args.result_parent_path)
    summary_writers.create_ppt(args.result_parent_path)
