import argparse
from pathlib import Path

from tenkit_tools.io.factor_export import export_best_components

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    parser.add_argument(
        "--is_single",
        type=bool,
        help="Whether subfolders should be iterated over.",
        default=False,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="mat",
        help="What file type, supported values are mat and hdf5 (or h5)",
    )
    args = parser.parse_args()

    # '/home/mariero/experiment_logs/MCIC/CP_ALS/CP_ALS_rank_2_01'

    if args.is_single:
        experiments = [Path(args.result_path)]
    else:
        experiments = sorted(
            filter(lambda x: x.is_dir(), Path(args.result_path).iterdir())
        )
    for experiment in experiments:
        print(f"Evaluating {experiment}")
        if not (experiment / "summaries" / "summary.json").is_file():
            print(f"Skipping {experiment}")
            continue
        out_name = f"{experiment.name}_factors." + args.dtype
        export_best_components(experiment, out_name=out_name)
