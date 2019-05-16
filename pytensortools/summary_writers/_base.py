import json


def load_summary(experiment_path):
    with (experiment_path/'summaries'/'summary.json').open() as f:
        return json.load(f)
    
def load_evaluations(experiment_path):
    with (experiment_path/'summaries'/'evaluations.json').open() as f:
        return json.load(f)

