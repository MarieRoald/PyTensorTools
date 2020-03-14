import multiprocessing
from pathlib import Path
from time import sleep

import h5py
import numpy as np
from scipy.io import loadmat

import tenkit_tools.experiment

NUM_THREADS = 64

DATA_PATH = Path("/home/mariero/datasets/CRNA/missing_060/datasets")
SAVE_PATH = Path("/work/mariero/experiment_logs/CRNA/first_smoothness/")

RANK = 2
MAX_ITS = 5000
NUM_RUNS = 20
START_DATASET = 0
END_DATASET = 100

RIDGE = 0.01
REGS = [0] + np.logspace(1, 7, 12).tolist()
SIMILARITY_IDS = ["freq_sigma_0_08_h5",  "freq_sigma_0_10_h5",  "freq_sigma_0_20_h5",]


def get_save_path(dataset_id, similarity_id):
    similarity_name = Path(similarity_id).stem

    return str(SAVE_PATH/similarity_name)


def get_experiment_params(dataset_id, similarity_id, reg):
    params = dict(
        experiment_name = f"{Path(dataset_id).stem}_reg_{reg:.02g}",
        save_path = get_save_path(dataset_id, similarity_id),
        num_runs = NUM_RUNS,
        num_processess = 1,
    )

    return params


def get_log_params():
    log_params = [
        {"type": "LossLogger"},
        {"type": "ExplainedVarianceLogger"}
    ]
    return log_params


def get_data_reader_params():
    data_reader_params = {
        "type": "MatlabDataReader",
        "arguments": {
            "file_path": str(DATA_PATH/"data_clean.mat"),
            "tensor_name": "data",
            "mode_names": ["node", "hour", "week", "month"],
        }
    }
    return data_reader_params


def load_mask(dataset_id):
    real_mask = ~loadmat(DATA_PATH/"data_clean.mat")["mask"]
    fake_mask = ~loadmat(DATA_PATH/dataset_id)["W_missing_dash"].astype(bool)

    return real_mask & fake_mask


def load_similarity(similarity_id):
    with h5py.File(DATA_PATH/"laplacians"/similarity_id, "r") as h5:
        return h5["laplacian"][:]


def get_decomposition_params(dataset_id, similarity_id, reg, ridge):
    similarity = load_similarity(similarity_id)
    mask = load_mask(dataset_id)

    return {
        "type": "CP_OPT", 
        "arguments": {
            "rank": RANK,
            "max_its": MAX_ITS,
            "checkpoint_frequency": 500,
            "convergence_tol": 1e-8,
            "lower_bounds": -np.inf,
            "upper_bounds": np.inf,
            "factor_constraints": [
                {'tikhonov_matrix': reg*similarity, 'ridge': ridge},
                {'ridge': ridge},
                {'ridge': ridge},
                {'ridge': ridge}
            ],
            
        },
        "fit_params": {"importance_weights": mask,}
    }


def run_experiment(experiment):
    ex = tenkit_tools.experiment.Experiment(**experiment)
    ex.run_experiments(should_print=False, parallel=False)


if __name__ == "__main__":
    experiments = []
    for similarity_id in SIMILARITY_IDS:
        for dataset_num in range(START_DATASET, END_DATASET):
            dataset_id = f"missing_060_{dataset_num+1:03d}.mat"
            Path(get_save_path(dataset_id, similarity_id)).mkdir(exist_ok=True, parents=True)
            for reg in REGS:
                experiments.append({
                    "experiment_params": get_experiment_params(dataset_id, similarity_id, reg),
                    "log_params": get_log_params(),
                    "data_reader_params": get_data_reader_params(),
                    "decomposition_params": get_decomposition_params(dataset_id, similarity_id, reg, RIDGE),
                    "preprocessor_params": []
                })
    
    # import tqdm 
    # tasks = [run_experiment(experiment) for experiment in tqdm.tqdm(experiments)]
    with multiprocessing.Pool(NUM_THREADS) as p:
        tasks = [p.apply_async(run_experiment, [experiment]) for experiment in experiments]

        prev_ready = 0
        num_ready = sum(task.ready() for task in tasks)
        while num_ready != len(tasks):
            if num_ready != prev_ready:
                print(f"{num_ready} out of {len(tasks)} completed tasks")
            prev_ready = num_ready
            num_ready = sum(task.ready() for task in tasks)
            sleep(0.5)
        results = [task.get() for task in tasks]
    print(results)
    
