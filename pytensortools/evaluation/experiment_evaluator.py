from pathlib import Path
import h5py

import json
import pytensor
from .. import datareader
from .. import evaluation

class ExperimentEvaluator:

    def __init__(self, single_run_evaluator_params=None, multi_run_evaluator_params=None):
        if single_run_evaluator_params is None:
            single_run_evaluator_params = []
        if multi_run_evaluator_params is None:
            multi_run_evaluator_params = []

        self.single_run_evaluators = self.create_evaluators(single_run_evaluator_params)
        self.multi_run_evaluators = self.create_evaluators(multi_run_evaluator_params)

    def create_evaluators(self, evaluator_params):
        evaluators = []
        for evaluator in evaluator_params:
            Evaluator = getattr(evaluation, evaluator['type'])
            arg = evaluator.get('arguments', {})
            evaluators.append(Evaluator(**arg))
        
        return evaluators

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
        results = {}
        with h5py.File(checkpoint_path) as h5:
            for run_evaluator in self.single_run_evaluators:
                results[run_evaluator.name] = run_evaluator._evaluate(data_reader, h5)
        # return the results as dict
        return results
    
    def load_data_reader_params(self, experiment_path):
        data_reader_path = experiment_path / 'parameters' / 'data_reader_params.json'
        with data_reader_path.open() as f:
            data_reader_params = json.load(f)
        return data_reader_params
        
    def generate_data_reader(self, data_reader_params):
        # TODO: This is copy-paste from experiment, should probably move somewhere else
        DataReader = getattr(datareader, data_reader_params['type'])
        return DataReader(**data_reader_params['arguments'])

    def evaluate_experiment(self, experiment_path):
        experiment_path = Path(experiment_path)
        # last inn summary fil
        summary = self.load_experiment_summary(experiment_path)
        
        data_reader_params = self.load_data_reader_params(experiment_path)
        data_reader = self.generate_data_reader(data_reader_params)

        best_run = summary['best_run']
        best_run_evaluations = self.evaluate_single_run(experiment_path, summary, data_reader)

        print(best_run_evaluations)
        # Last inn all runs???
        
        

        # kjør multi_run_evaluators på alle?
        
        # kjør single run evaluators på beste run?
        pass