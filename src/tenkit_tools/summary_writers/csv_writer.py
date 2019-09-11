import csv
import os
from collections import ChainMap
from pathlib import Path

from ..utils import load_evaluations, load_summary

CSV_FILE = "slide.csv"


def _format_csv_rank(rank):
    return f"{rank:d}"


def _format_csv_core_consistency(best_run_evaluations):
    core_consistency = best_run_evaluations.get("Core Consistency", "-")
    if core_consistency == "-":
        core_consistency = best_run_evaluations.get("Parafac2 Core Consistency", "-")
    if isinstance(core_consistency, float) or isinstance(core_consistency, int):
        if core_consistency < 0:
            return "<0"
        else:
            return f"{core_consistency:.0f}"
    return core_consistency


def _format_csv_pvalue(best_run_evaluations):
    pval = best_run_evaluations.get("Best P value for mode 0", "-")
    if pval == "-":
        pval = best_run_evaluations.get("Best P value for mode 2", "-")
    if isinstance(pval, float) or isinstance(pval, int):
        pval = f"{pval:.1e}"
    return pval


def _format_csv_explained(best_run_evaluations):
    explained = best_run_evaluations.get("Explained variance", "-")
    if isinstance(explained, float) or isinstance(explained, int):
        explained = f"{int(100*explained):2d}%"
    return explained


def _format_csv_clustering(best_run_evaluations):
    clustering = best_run_evaluations.get("Max Kmeans clustering accuracy", "-")
    if isinstance(clustering, float) or isinstance(clustering, int):
        clustering = f"{int(100*clustering):2d}%"
    return clustering


def _format_csv_row(experiment_path, summary, best_run_evaluations):
    rank = _format_csv_rank(summary["model_rank"])
    core_consistency = _format_csv_core_consistency(best_run_evaluations)
    pval = _format_csv_pvalue(best_run_evaluations)
    explained = _format_csv_explained(best_run_evaluations)
    clustering = _format_csv_clustering(best_run_evaluations)
    return {
        "Number of Components": rank,
        "Core Consistency": core_consistency,
        r"% Explained": explained,
        "Significant Factors (p-values)": pval,
        "Clustering (max acc)": clustering,
    }


def _write_csv_row(experiment_path, csvpath=None):
    summary = load_summary(experiment_path)
    best_run_evaluations = load_evaluations(experiment_path)["best_run_evaluations"]
    # TODO: Maybe do something about this somewhere else?
    best_run_evaluations = dict(ChainMap(*best_run_evaluations))

    if csvpath is None:
        csvpath = experiment_path.parent / CSV_FILE

    writeheader = False
    if not csvpath.is_file():
        writeheader = True

    with csvpath.open("a") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Number of Components",
                "Core Consistency",
                r"% Explained",
                "Significant Factors (p-values)",
                "Clustering (max acc)",
            ],
        )
        if writeheader:
            writer.writeheader()
        writer.writerow(_format_csv_row(experiment_path, summary, best_run_evaluations))


def create_csv(experiment_parent, new_file=False):
    if new_file and (Path(experiment_parent) / CSV_FILE).is_file():
        os.remove(experiment_parent / CSV_FILE)
    for experiment in experiment_parent.iterdir():
        if not (experiment / "summaries/summary.json").is_file():
            continue
        if experiment.is_dir():
            _write_csv_row(experiment)
