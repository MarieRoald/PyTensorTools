import argparse
import json
import sys
from pathlib import Path

from tenkittools.experiment import Experiment


# TODO: Add argument to assert that decomposer params is set

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_path",
        help="Path to a folder containing the following files:\n"
        "\tdata_reader_params.json\n"
        "\tlog_params.json\n"
        "\tdecomposition_params.json\n",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        help="The number of runs to perform, default=60",
        type=int,
        default=60,
    )
    parser.add_argument(
        "-s",
        "--save_path",
        help="The location to store the experiment results, default='.'",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-r", "--rank", help="The rank of the decomposition", type=int, default=None
    )
    parser.add_argument(
        "--tol", help="The convergence tolerance", type=float, default=None
    )
    parser.add_argument(
        "--max_its",
        help="The maximum number of iterations to run.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--min_rank",
        help="The minimum number of components to fit. Cannot be set simultaneously with --rank and requires --max_rank",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_rank",
        help="The minimum number of components to fit. Cannot be set simultaneously with --rank and requires --min_rank",
        type=int,
        default=None,
    )
    parser.add_argument("--rank_step", type=int, default=1)
    parser.add_argument("--load_id", type=int, default=None)
    # TODO: Move this to separate file
    parser.add_argument(
        "--save_dataset",
        help="Save the dataset file in the log folder. If this is set, then the experiments will not be run.",
    )
    parser.add_argument(
        "--dataset_labels",
        help="The name of the classes that should be regarded as labels. Separate label names by comma and modes by underscore",
        default=[],
    )
    parser.add_argument(
        "--save_preprocessed",
        help="Whether the dataset should be processed before saving",
        default=False,
    )
    args = parser.parse_args()

    # Load experiment params
    experiment_path = Path(args.experiment_path)
    with (experiment_path / "data_reader_params.json").open() as f:
        data_reader_params = json.load(f)
    with (experiment_path / "log_params.json").open() as f:
        logger_params = json.load(f)
    with (experiment_path / "decomposition_params.json").open() as f:
        decomposition_params = json.load(f)
    preprocessor_params = None
    if (experiment_path / "preprocessor_params.json").is_file():
        with (experiment_path / "preprocessor_params.json").open() as f:
            preprocessor_params = json.load(f)

    # Modify experiment params according to input flags
    if args.save_path is not None:
        save_path = args.save_path
    else:
        with (experiment_path / "save_path.json").open() as f:
            save_path = json.load(f)
            if "save_path" in save_path:
                save_path = save_path["save_path"]
            else:
                save_path = "."

    if args.rank is not None:
        decomposition_params["arguments"]["rank"] = args.rank
        if args.min_rank is not None or args.max_rank is not None:
            raise ValueError(
                "Cannot set rank and rank-range (min/max_rank) simultaneously."
            )
    if args.min_rank is not None and args.max_rank is not None:
        run_single = False
    elif args.min_rank is not None or args.max_rank is not None:
        raise ValueError("Both rank and rank-range must be set")

    if args.max_its is not None:
        decomposition_params["arguments"]["max_its"] = args.max_its

    if args.tol is not None:
        decomposition_params["arguments"]["convergence_tol"] = args.tol

    decomposition_type = decomposition_params["type"]
    experiment_params = {
        "num_runs": args.num_runs,
        "save_path": str(save_path),
        "experiment_name": experiment_path.name,
    }

    # Run experiment
    if args.save_dataset:
        experiment = Experiment(
            experiment_params,
            data_reader_params,
            decomposition_params,
            logger_params,
            preprocessor_params=preprocessor_params,
        )
        label_names = [
            [l for l in labels.split(",") if l != ""]
            for labels in args.dataset_labels.split("_")
        ]
        dataset_parent = (
            Path(experiment_params["save_path"]) / experiment_params["experiment_name"]
        )
        if args.save_preprocessed:
            experiment.data_reader.to_matlab(
                label_names, dataset_parent / "preprocessed_dataset.mat"
            )
            print("Saved preprocessed")
        else:
            experiment.generate_data_reader().to_matlab(
                label_names, dataset_parent / "dataset.mat"
            )
    elif run_single:
        experiment = Experiment(
            experiment_params,
            data_reader_params,
            decomposition_params,
            logger_params,
            preprocessor_params=preprocessor_params,
            load_id=args.load_id,
        )
        runs = experiment.run_experiments()
    else:
        for rank in range(args.min_rank, args.max_rank, args.rank_step):
            decomposition_params["arguments"]["rank"] = rank
            experiment = Experiment(
                experiment_params,
                data_reader_params,
                decomposition_params,
                logger_params,
                preprocessor_params=preprocessor_params,
                load_id=args.load_id,
            )
            runs = experiment.run_experiments()
