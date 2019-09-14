import csv
import json
from collections import ChainMap
from pathlib import Path
from typing import Dict

import h5py
import xlsxwriter

import tenkit

from .. import datareader, evaluation, preprocessor, utils
from .single_run_visualisers import create_visualisers
from .base_evaluator import create_evaluators


class ExperimentEvaluator:
    def __init__(
        self,
        data_reader=None,
        single_run_evaluator_params=None,
        multi_run_evaluator_params=None,
        single_run_visualiser_params=None,
        postprocessor_params=None,
    ):
        if single_run_evaluator_params is None:
            single_run_evaluator_params = []
        if multi_run_evaluator_params is None:
            multi_run_evaluator_params = []
        if single_run_visualiser_params is None:
            single_run_visualiser_params = []

        self.single_run_evaluator_params = single_run_evaluator_params
        self.multi_run_evaluator_params = multi_run_evaluator_params
        self.single_run_visualiser_params = single_run_visualiser_params
        self.postprocessor_params = postprocessor_params

    def evaluate_single_run(
        self,
        experiment_path: Path,
        summary: dict,
        data_reader: datareader.BaseDataReader,
    ) -> dict:
        # TODO: maybe have the possibility of evaluating other run than best run?
        checkpoint_path = experiment_path / "checkpoints" / summary["best_run"]

        single_run_evaluators = create_evaluators(
            self.single_run_evaluator_params,
            summary,
            postprocessor_params=self.postprocessor_params,
            data_reader=data_reader,
        )

        results = []
        with h5py.File(checkpoint_path) as h5:
            for run_evaluator in single_run_evaluators:
                results.append(run_evaluator._evaluate(data_reader, h5))
        # return the results as dict
        return results

    def visualise_single_run(
        self,
        experiment_path: Path,
        summary: dict,
        data_reader: datareader.BaseDataReader,
    ) -> dict:
        # TODO: maybe have the possibility of evaluating other run than best run?
        checkpoint_path = experiment_path / "checkpoints" / summary["best_run"]

        single_run_visualisers = create_visualisers(
            self.single_run_visualiser_params,
            summary,
            postprocessor_params=self.postprocessor_params,
            data_reader=data_reader,
        )

        results = {}
        figure_path = experiment_path / "summaries" / "visualizations"
        if not figure_path.is_dir():
            figure_path.mkdir()

        with h5py.File(checkpoint_path) as h5:
            for run_visualiser in single_run_visualisers:
                results[run_visualiser.name] = run_visualiser._visualise(
                    data_reader, h5
                )
                # TODO:skal dette skje her?
                results[run_visualiser.name].savefig(
                    figure_path / f'{run_visualiser.name}_{summary["best_run"]}.png'
                )
        # return the results as dict
        return results

    def evaluate_multiple_runs(self, experiment_path, summary, data_reader):
        checkpoint_path = Path(experiment_path) / "checkpoints"
        multi_run_evaluators = create_evaluators(
            self.multi_run_evaluator_params,
            summary,
            postprocessor_params=self.postprocessor_params,
            data_reader=data_reader,
        )
        results = {}

        for run_evaluator in multi_run_evaluators:
            results[run_evaluator.name] = run_evaluator(data_reader, checkpoint_path)

        return results

    def evaluate_experiment(self, experiment_path):
        experiment_path = Path(experiment_path)

        # Load info from experiment
        summary = utils.load_summary(experiment_path)

        experiment_params = utils.load_experiment_params(experiment_path)
        data_reader_params = experiment_params["data_reader_params"]
        preprocessor_params = experiment_params["preprocessor_params"]

        data_reader = utils.generate_data_reader(data_reader_params, preprocessor_params)

        # Evaluate
        best_run_evaluations = self.evaluate_single_run(
            experiment_path, summary, data_reader
        )
        print(best_run_evaluations)

        best_run_visualisations = self.visualise_single_run(
            experiment_path, summary, data_reader
        )
        print(best_run_visualisations)

        multi_run_evaluations = self.evaluate_multiple_runs(
            experiment_path, summary, data_reader
        )
        print(multi_run_evaluations)

        with (experiment_path / "summaries" / "evaluations.json").open("w") as f:
            json.dump(
                {
                    "best_run_evaluations": best_run_evaluations,
                    "multi_run_evaluations": multi_run_evaluations,
                },
                f,
            )

