#!/home/marie/anaconda3/bin/python
import argparse
import json
import sys
from pathlib import Path

import tenkit.base
from tenkit_tools import summary_writers
from tenkit_tools.evaluation.experiment_evaluator import ExperimentEvaluator
from tenkit_tools.experiment import Experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    parser.add_argument("evaluator_params", type=str)
    parser.add_argument(
        "--is_single",
        type=bool,
        help="Whether subfolders should be iterated over.",
        default=False,
    )
    parser.add_argument("--skip_evaluation", default=False)
    args = parser.parse_args()

    with open(args.evaluator_params) as f:
        evaluator_params = json.load(f)

    evaluator = ExperimentEvaluator(**evaluator_params)
    # '/home/mariero/experiment_logs/MCIC/CP_ALS/CP_ALS_rank_2_01'

    experiments = sorted(filter(lambda x: x.is_dir(), Path(args.result_path).iterdir()))
    if args.is_single:
        experiments = [Path(args.result_path)]

    if not args.skip_evaluation:
        for experiment in sorted(experiments):
            print(f"Evaluating {experiment}")
            if not (experiment / "summaries" / "summary.json").is_file():
                print(f"Skipping {experiment}")
                continue
            evaluator.evaluate_experiment(str(experiment))
            summary_writers.create_spreadsheet(Path(experiment))

    if not args.is_single:
        summary_writers.create_csv(Path(args.result_path), new_file=True)
        summary_writers.create_ppt(Path(args.result_path))
