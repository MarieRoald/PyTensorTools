from pathlib import Path
import argparse
import json
from tenkit_tools.evaluation.experiment_evaluator import ExperimentEvaluator
from csv import DictWriter
from scipy.io import loadmat
from tenkit_tools.utils import load_summary
from tqdm import tqdm


DATA_PATH = Path("/home/mariero/datasets/CRNA/missing_060/datasets")
SIMILARITY_IDS = ["freq_sigma_0_08_h5",  "freq_sigma_0_10_h5",  "freq_sigma_0_20_h5",]
DATASET_PATTERN = "missing_{missing_percent:03d}_{dataset_id:03d}.mat"


def evaluate_experiment(
    experiment_info: dict,
    attempt_num: int,
    experiment_subfolder: Path,
    evaluator: ExperimentEvaluator
):
    # print(f"Evaluating {experiment_subfolder}")	
    if not (experiment_subfolder / "summaries" / "summary.json").is_file():
        return None
    
    summary = load_summary(experiment_subfolder)
    for summary_entry in ["best_run", "best_fit", "best_loss", "std_loss", "std_fit"]:
        experiment_info[summary_entry] = summary[summary_entry]

    eval_results, _ = evaluator.evaluate_experiment(str(experiment_subfolder),verbose=False)

    experiment_info['attempt_num'] = i
    for metric in eval_results:
        experiment_info.update(metric)
    return experiment_info


def get_experiment_info(experiment_folder: Path):
    _, missing_percentage, dataset_id, _, reg = experiment_folder.name.split("_")

    experiment = {}
    experiment['experiment_name'] = f'{experiment_folder.parent.stem}_{experiment_folder.stem}'
    experiment['similarity'] = experiment_folder.parent.stem
    experiment['missing_percentage'] = int(missing_percentage)
    experiment['dataset_id'] = int(dataset_id)
    experiment['smoothness_reg'] = float(reg)

    return experiment


def load_mask(dataset_id):
    #real_mask = ~loadmat(DATA_PATH/"data_clean.mat")["mask"]
    fake_mask = ~loadmat(DATA_PATH/dataset_id)["W_missing_dash"].astype(bool)

    return fake_mask


def get_extra_evaluators(missing_percent, dataset_id):
    mask_filename = DATASET_PATTERN.format(
        missing_percent=missing_percent,
        dataset_id=dataset_id
    )
    mask = load_mask(mask_filename) 

    return [{
        "type":"TensorCompletionScore",
        "arguments": {
            "importance_weights": mask
        }
    }]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_folder", type=str)
    parser.add_argument("evaluator_params", type=str)

    args = parser.parse_args()

    experiments_folder = Path(args.experiments_folder)

    with open(args.evaluator_params) as f:
        evaluator_params = json.load(f)


    num_rows = 0
    # The outer loop iterates over the folders corresponding to different similarity matrices
    for single_similarity_experiments_folder in sorted(experiments_folder.iterdir()):
        print(single_similarity_experiments_folder.stem)
        # Within each of these folders we have a series of experiments with different datasets
        # and regularisation strengths. This loop iterates over these experiments.
        for experiment_folder in tqdm(sorted(single_similarity_experiments_folder.iterdir())):
            experiment_info = get_experiment_info(experiment_folder)
            # print(f'Loading experiment: {experiment_info["experiment_name"]}')

            evaluator_params['single_run_evaluator_params'].extend(
                get_extra_evaluators(
                    missing_percent=experiment_info['missing_percentage'],
                    dataset_id=experiment_info['dataset_id']
                )
            )
            evaluator = ExperimentEvaluator(**evaluator_params)

            # Finally, each experiment may have been attempted several times.
            # For example with different numbers of components.
            folders = sorted(filter(lambda x: x.is_dir(), Path(experiment_folder).iterdir()))
            for i, experiment_subfolder in enumerate(folders):
                experiment_row = experiment_info.copy()

                # Set rank
                rank = int(experiment_subfolder.name.split("rank_")[1].split("_")[0])
                experiment_row["rank"] = rank

                # Evaluate the experiment
                experiment_row = evaluate_experiment(
                    experiment_row, i, experiment_subfolder, evaluator
                )
                if experiment_row is None:
                    print(f"Skipping {experiment_subfolder}")
                    continue

                open_mode = 'w' if num_rows == 0 else 'a'
                # Write summary csv
                with (experiments_folder/'results.csv').open(open_mode) as f:
                    writer = DictWriter(f, fieldnames=list(experiment_row.keys()))
                    if num_rows == 0:
                        writer.writeheader()
                    writer.writerow(experiment_row)
                    num_rows += 1

