import argparse
from pathlib import Path
import tenkittools.summary_writers as summary_writers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_parent')
    args = parser.parse_args()

    experiment_parent = Path(args.experiment_parent)
    experiments = sorted(
        filter(
            lambda x: x.is_dir(), experiment_parent.iterdir()
        )
    )
    for experiment_path in experiments:
        print(experiment_path)
        if not (experiment_path/'summaries'/'evaluations.json').is_file():
             print('skipping', experiment_path)
             continue
        summary_writers.create_spreadsheet(experiment_path)
    
    summary_writers.create_csv(experiment_parent)
    summary_writers.create_ppt(experiment_parent)
