from pathlib import Path
from typing import Dict
import h5py
import xlsxwriter

import json
import tenkit

from .. import datareader
from .. import evaluation
from .. import preprocessor
from .base_evaluator import create_evaluators
from ..visualization.base_visualiser import create_visualisers

from collections import ChainMap
import csv


class ExperimentEvaluator:
    def __init__(
        self,
        single_run_evaluator_params=None,
        multi_run_evaluator_params=None,
        single_run_visualiser_params=None
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

    def load_experiment_summary(self, experiment_path):
        summary_path = experiment_path / 'summaries/summary.json'

        with summary_path.open('r') as f:
            summary = json.load(f)
        return summary
        
    def evaluate_single_run(
        self,
        experiment_path : Path,
        summary : dict,
        data_reader : datareader.BaseDataReader
    ) -> dict:
        # TODO: maybe have the possibility of evaluating other run than best run?
        checkpoint_path = experiment_path / 'checkpoints'/ summary['best_run']

        #decomposer = model_type(rank=model_rank, init_scheme='from_checkpoint')
        #decomposer._init_fit(data_reader.tensor, initial_decomposition=checkpoint_path)
        single_run_evaluators = create_evaluators(self.single_run_evaluator_params, summary)

        results = []
        with h5py.File(checkpoint_path) as h5:
            for run_evaluator in single_run_evaluators:
                results.append(run_evaluator._evaluate(data_reader, h5))
        # return the results as dict
        return results

    def visualise_single_run(
        self,
        experiment_path : Path,
        summary : dict,
        data_reader : datareader.BaseDataReader
    ) -> dict:
        # TODO: maybe have the possibility of evaluating other run than best run?
        checkpoint_path = experiment_path / 'checkpoints'/ summary['best_run']

        single_run_visualisers = create_visualisers(self.single_run_visualiser_params, summary)

        results = {}
        figure_path = experiment_path/'summaries'/'visualizations'
        if not figure_path.is_dir():
            figure_path.mkdir()
    
        with h5py.File(checkpoint_path) as h5:
            for run_visualiser in single_run_visualisers:
                results[run_visualiser.name] = run_visualiser._visualise(data_reader, h5)
                #TODO:skal dette skje her?
                results[run_visualiser.name].savefig(figure_path/f'{run_visualiser.name}_{summary["best_run"]}.png')
        # return the results as dict
        return results
    
    def load_data_reader_params(self, experiment_path):
        data_reader_path = experiment_path / 'parameters' / 'data_reader_params.json'
        with data_reader_path.open() as f:
            data_reader_params = json.load(f)

        preprocessor_params = None
        preprocessor_params_path = experiment_path / 'parameters' / 'preprocessor_params.json'

        if preprocessor_params_path.is_file():
            with preprocessor_params_path.open() as f:
                preprocessor_params = json.load(f)
        return data_reader_params, preprocessor_params
        
    def preprocess_data(self, data_reader, preprocessor_params):
        # This should not be a method inside a class... a copy
        if preprocessor_params is not None:
            if isinstance(preprocessor_params, Dict):
                preprocessor_params = [preprocessor_params]
            
            for preprocessor_params in preprocessor_params:
                Preprocessor = getattr(preprocessor, preprocessor_params['type'])
                args = preprocessor_params.get('arguments', {})
                data_reader = Preprocessor(data_reader=data_reader, **args)
        return data_reader

    def generate_data_reader(self, data_reader_params, preprocessor_params):
        # TODO: This is copy-paste from experiment, should probably move somewhere else
        DataReader = getattr(datareader, data_reader_params['type'])
        data_reader = DataReader(**data_reader_params['arguments'])
        data_reader = self.preprocess_data(data_reader, preprocessor_params)

        return data_reader
    
    def evaluate_multiple_runs(self, experiment_path, summary, data_reader):
        checkpoint_path = Path(experiment_path)/'checkpoints'
        multi_run_evaluators = create_evaluators(self.multi_run_evaluator_params, summary)
        results = {}

        for run_evaluator in multi_run_evaluators:
            results[run_evaluator.name] = run_evaluator(data_reader, checkpoint_path)
        
        return results


    def evaluate_experiment(self, experiment_path):
        experiment_path = Path(experiment_path)
        # last inn summary fil
        summary = self.load_experiment_summary(experiment_path)
        
        data_reader_params, preprocessor_params = self.load_data_reader_params(experiment_path)
        data_reader = self.generate_data_reader(data_reader_params, preprocessor_params)


        best_run = summary['best_run']
        best_run_evaluations = self.evaluate_single_run(experiment_path, summary, data_reader)
        best_run_visualisations = self.visualise_single_run(experiment_path, summary, data_reader)

        print(best_run_evaluations)

        multi_run_evaluations = self.evaluate_multiple_runs(experiment_path, summary, data_reader)
        print(multi_run_evaluations)
        # Last inn all runs???
        
        with (experiment_path/'summaries'/'evaluations.json').open('w') as f:
            json.dump({'best_run_evaluations': best_run_evaluations, 'multi_run_evaluations': multi_run_evaluations}, f)

        # kjør multi_run_evaluators på alle?
        
        # kjør single run evaluators på beste run?
        pass
