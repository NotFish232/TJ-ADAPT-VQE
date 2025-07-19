import json
import re
from functools import lru_cache, reduce
from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from matplotlib.figure import Figure
from mlflow.entities import Run
from mlflow.store.entities import PagedList
from mlflow.tracking import MlflowClient
from typing_extensions import Any

from .observables import NumberObservable, SpinSquaredObservable, SpinZObservable
from .optimizers import LBFGSOptimizer

# from .pools import FSDPool, GSDPool, QEBPool, UnresIndividualTUPSPool
from .utils.molecules import Molecule

RUN_DIR = "./runs/"
mlflow.set_tracking_uri(RUN_DIR)

RESULTS_DIR = "./results"


CAPITALIZATION_RULES = [
    ("tups", "tUPS"),
    ("ucc", "UCC"),
    ("fsd", "FSD"),
    ("gsd", "GSD"),
    ("qeb", "QEB"),
]


@lru_cache(maxsize=None)
def get_runs() -> PagedList[Run]:
    """
    Simple wrapper around fetching all runs that allows it to be lru_cache'd.

    Returns:
        PagedList[Run]: All runs
    """
    client = MlflowClient()

    return client.search_runs(experiment_ids=["0"])


@lru_cache(maxsize=None)
def get_run_params(run_id: str) -> dict[str, Any]:
    """
    Retrieves all parameters associated with a mlflow run ID.
    Performs the necessary processing on those params

    Args:
        run_id (str): The MLflow run ID.

    Returns:
        dict[str, list[tuple[int, float]]]: dictionary mapping each metric name
        to a sorted list of (step, value) tuples.

    """
    client = MlflowClient()
    raw_params = client.get_run(run_id).data.params

    params = {}

    for param in raw_params:
        params[param] = raw_params[param]

        # convert json compatible params into json
        try:
            params[param] = json.loads(params[param])
        except ValueError:
            pass

    params["starting_ansatz"] = " ".join(
        ansatz["_name"] for ansatz in params["starting_ansatz"]
    )
    if "pool" not in params:
        params["pool"] = {"name": params["starting_ansatz"][1]}

    return params


@lru_cache(maxsize=None)
def get_run_metrics(run_id: str) -> dict[str, Any]:
    """
    Retrieves all logged metric histories from a given MLflow run ID.
    Each metric's values are sorted by step to help with plotting.

    Args:
        run_id (str): The MLflow run ID.

    Returns:
        dict[str, list[tuple[int, float]]]: dictionary mapping each metric name
        to a sorted list of (step, value) tuples.

    """
    client = MlflowClient()
    raw_metrics = client.get_run(run_id).data.metrics

    metrics = {}

    for metric in raw_metrics:
        history = client.get_metric_history(run_id, metric)
        sorted_history = sorted(
            [(h.step, h.value) for h in history], key=lambda x: x[0]
        )
        metrics[metric] = sorted_history

    return metrics


def check_filtered(
    params: dict[str, Any], filter_fixed: dict[str, Any], filter_ignored: dict[str, Any]
) -> bool:
    """
    Check if run should be filtered away or not. Implements the logic used for filter_fixed / filter_ignored.

    Args:
        params (dict[str, Any]): The params of the run.
         filter_fixed (dict[str, Any]): Dictionary of fixed parameters to filter by.
        filter_ignored (dict[str, list[Any]]): Dictionary of parameters that if satisifed should be ignored.

    Returns:
        bool: If the run should be filtered away.
    """

    # filter for fixed values
    for key, val in filter_fixed.items():
        if get_nested_json(params, key) != val:
            return True

    # Filter for ignored values
    for key, val in filter_ignored.items():
        if get_nested_json(params, key) in val:
            return True

    return False


def get_nested_json(data: dict[str, Any], key: str) -> Any:
    """
    Extracts key from nested dictionary, where . in key signifies a break between different actual
    key pairs.

    Args:
        data (dict[str, Any]): The dictionary of data.
        key (str): The key with parts seperated by '.'.

    Returns:
        Any: The result key or None if not exists.
    """

    return reduce(lambda x, y: None if x is None else x.get(y), key.split("."), data)  # type: ignore


def adjust_capitalization(s: str) -> str:
    """
    Replaces all values in s with the associated properly capitalized version.
    I.e. tups shoulld be tUPS. s should be lowercase.

    Args:
        s (str): String to replace elements of.

    Returns:
        str: the Formatted str.
    """

    s = " ".join(x.capitalize() for x in s.split("_"))

    for t, ct in CAPITALIZATION_RULES:
        s = re.compile(t, re.IGNORECASE).sub(ct, s)

    return s


def compare_runs(
    *,
    group_by: str,
    x_parameter: str | None = None,
    y_parameter: str,
    title: str = "",
    x_axis_title: str = "",
    y_axis_title: str = "",
    filter_fixed: dict[str, Any] = {},
    filter_ignored: dict[str, list[Any]] = {},
    log_scale: bool = False,
    adapt_bars: bool = False,
    truncate_runs: bool = False,
) -> Figure:
    """
    Comparing multiple runs grouped by a specified parameter, fixed by a specific filter, and with specific x and y axis.

    Args:
        group_by (str): Parameter name to group runs by (e.g., "optimizer"). Dependent Variable.
        x_parameter (str | None, optional): The parameter for the x axis. Defaults to None.
        y_parameter (str): The parameter to actually plot on the graph, like energy_percent.
        title: (str, optional): The title for the graph. Defaults to "".
        x_axis_title: (str, optional): The x axis title. Defaults to "".
        y_axis_title: (str, optional): The y axis title. Defaults to "".
        filter_fixed (dict[str, Any], optional): Dictionary of fixed parameters to filter by. The constant stuff. Defaults to {}.
        filter_ignored (dict[str, list[Any]], optional): Dictionary of parameters that if satisifed should be ignored. Useful if you want
        to remove some runs from a plot while keeping others. Defaults to {}.
        log_scale (bool, optional): Whether to use log scale on y axis. Defaults to False.
        adapt_bars (bool, optional): Whether or not to display verticla bars representing adapt iterations. Defaults to False.
        truncate_runs (bool, optional): Whether to truncate exceptionally long runs to limit horizontal size of plot. Defaults to False.

    Returns:
        Figure: Matplotlib plot.
    """
    runs = get_runs()

    grouped_runs = {}  # type: ignore

    for run in runs:
        run_id = run.info.run_id
        params = get_run_params(run_id)

        # check whether to use or skip this run
        if check_filtered(params, filter_fixed, filter_ignored):
            continue

        # Group by selected parameter
        if (group_val := get_nested_json(params, group_by)) is not None:
            grouped_runs.setdefault(group_val, []).append(run_id)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(19.20, 10.80))

    # technically you want a min of the energy and a max of the iter but it doesn't matter since floats are never equal
    if grouped_runs and truncate_runs:
        max_iter = min(
            get_run_metrics(run_id)[y_parameter][-1][::-1]
            for run_ids in grouped_runs.values()
            for run_id in run_ids
        )[1]

    # list of tuples of (x, y, color)
    error_bars = []

    for group, run_ids in sorted(grouped_runs.items()):
        for run_id in run_ids:
            metrics = get_run_metrics(run_id)
            if y_parameter not in metrics:
                continue

            x_vals, y_vals = zip(*metrics[y_parameter])

            if x_parameter is not None:
                if x_parameter not in metrics:
                    continue

                t_vals, x_vals = zip(*metrics[x_parameter])

                # x_vals is smaller than y_vals shrink y_vals based on the time stamps in t_vals
                # since the shorter x values are taken at the start of each adapt vqe iteration, we need to fast forward to the end
                if len(x_vals) < len(y_vals):
                    t_vals = [t - 1 for t in t_vals[:-1]] + [len(y_vals) - 1]  # type: ignore
                    y_vals = [y_vals[t] for t in t_vals]  # type: ignore

            if truncate_runs:
                x_vals, y_vals = x_vals[: max_iter + 5], y_vals[: max_iter + 5]

            l = plt.plot(x_vals, y_vals, marker="o", label=adjust_capitalization(group))

            if adapt_bars and x_parameter is None:
                color = l[0].get_color()

                # plot error bar for each adapt iteration if x parameter is just time
                for i, (t, n_param) in enumerate(metrics["n_params"]):
                    if truncate_runs and t - 1 >= max_iter + 5:
                        break

                    if i == 0 or n_param != metrics["n_params"][i - 1]:
                        x_i = x_vals[t - 1]
                        y_i = y_vals[t - 1]

                    error_bars.append((x_i, y_i, color))

    if y_parameter == "energy_percent":  # plot chemical accuracy
        plt.axhline(y=0.00159, color="gray", linestyle="--")
        plt.axhspan(
            ax.get_ylim()[0],
            0.00159,
            color="gray",
            alpha=0.25,
            label="Chemical Accuracy",
        )

    plt.title(title, fontsize=32)
    plt.xlabel(x_axis_title, fontsize=29)
    plt.ylabel(y_axis_title, fontsize=29)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=29, loc="upper left")

    if log_scale:
        ax.set_yscale("log")

    # add error bars to the graph
    _, y0, _, y1 = ax.viewLim.bounds
    for x_i, y_i, color in error_bars:
        new_color = "#" + "".join(
            hex(min(15, (ord(c) - 48) % 39 + 4))[2] for c in color[1:]  # type: ignore
        )

        if log_scale:
            y_norm = abs(log(y_i / abs(y0), y1 / y0))
        else:
            y_norm = (y_i - y0) / (y1 - y0)

        plt.axvline(x_i, ymin=y_norm - 0.05, ymax=y_norm + 0.05, color=new_color)

    return fig


def main() -> None:
    molecules = [
        Molecule.H2,
        Molecule.H2_631G,
        Molecule.H3,
        Molecule.H4,
        Molecule.H5,
        Molecule.LiH,
    ]

    optimizers = [LBFGSOptimizer]

    # pools = [FSDPool, GSDPool, QEBPool, UnresIndividualTUPSPool]

    observables = [NumberObservable, SpinZObservable, SpinSquaredObservable]
    metrics = ["n_params", "circuit_depth", "cnot_count"]

    for m_t in molecules:
        molecule = m_t(1, False)
        molecule_dir = f"{RESULTS_DIR}/{molecule.name}_{molecule.basis}"

        # graphs for pools
        for optimizer in optimizers:
            fig = compare_runs(
                group_by="pool._name",
                y_parameter="energy_percent",
                title=f"Energy Error with {adjust_capitalization(optimizer._name())} on {molecule.name} ({molecule.basis})",
                x_axis_title="Cumulative VQE Iterations",
                y_axis_title="Energy Error in a.u",
                filter_fixed={
                    "optimizer._name": optimizer._name(),
                    "qiskit_backend.shots": 0,
                    "molecule.name": molecule.name,
                    "molecule.basis": molecule.basis,
                },
                log_scale=True,
                adapt_bars=True,
                truncate_runs=True,
            )

            Path(f"{molecule_dir}/pools").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{molecule_dir}/pools/{optimizer._name()}.png")
            plt.close(fig)

        # for each other metric
        for metric in metrics:
            fig = compare_runs(
                group_by="pool._name",
                y_parameter=metric,
                title=f"{adjust_capitalization(metric)} with LBFGS on {molecule.name} ({molecule.basis})",
                x_axis_title="Cumulative VQE Iterations",
                y_axis_title=adjust_capitalization(metric),
                filter_fixed={
                    "optimizer._name": optimizer._name(),
                    "qiskit_backend.shots": 0,
                    "molecule.name": molecule.name,
                    "molecule.basis": molecule.basis,
                },
            )

            Path(f"{molecule_dir}/pools").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{molecule_dir}/pools/{metric}.png")
            plt.close(fig)

        # special one for CNOT count as the x parameter and energy_percent as the y variable
        fig = compare_runs(
            group_by="pool._name",
            x_parameter="cnot_count",
            y_parameter="energy_percent",
            title=f"Energy Error by CNOT count with LBFGS on {molecule.name} ({molecule.basis})",
            x_axis_title="CNOT Count",
            y_axis_title="Energy Error in a.u",
            filter_fixed={
                "optimizer._name": LBFGSOptimizer._name(),
                "qiskit_backend.shots": 0,
                "molecule.name": molecule.name,
                "molecule.basis": molecule.basis,
            },
            log_scale=True,
            truncate_runs=True,
        )
        Path(f"{molecule_dir}/pools").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{molecule_dir}/pools/cnot_x.png")
        plt.close(fig)

        # graphs for observables
        for observable in observables:
            fig = compare_runs(
                group_by="pool._name",
                y_parameter=observable._name(),
                title=f"{adjust_capitalization(observable._name())} with LBFGS on {molecule.name} ({molecule.basis})",
                x_axis_title="Cumulative VQE Iterations",
                y_axis_title="Expectation value",
                filter_fixed={
                    "optimizer._name": optimizer._name(),
                    "qiskit_backend.shots": 0,
                    "molecule.name": molecule.name,
                    "molecule.basis": molecule.basis,
                },
            )

            Path(f"{molecule_dir}/observables").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{molecule_dir}/observables/{observable._name()}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
