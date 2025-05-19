import json
import re
from functools import reduce
from itertools import product
from math import log10
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from typing_extensions import Any

RUN_DIR = "./runs/"
mlflow.set_tracking_uri(RUN_DIR)

OUT_DIR = "./results"


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


def get_logged_metrics(run_id: str):
    """
    Retrieves all logged metric histories from a given MLflow run ID.
    Each metric's values are sorted by step to help with plotting.

    Args:
        run_id (str): The MLflow run ID.

    Returns:
        dict[str, list[tuple[int, float]]]: dictionary mapping each metric name
        to a sorted list of (step, value) tuples.

    Currently found metrics: ['n_params', 'adapt_energy', 'number_observable', 'avg_grad',
    'energy', 'energy_percent', 'spin_squared_observable', 'energy_percent_log',
    'adapt_operator_grad', 'max_grad', 'adapt_operator_idx', 'spin_z_observable']
    """
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data

    metrics = {}

    for k in data.metrics:
        history = client.get_metric_history(run_id, k)
        sorted_history = sorted(
            [(h.step, h.value) for h in history], key=lambda x: x[0]
        )
        metrics[k] = sorted_history

    # some metrics have to be processed

    # make n params, cnot count, and circuit depth same length as energy
    for metric in ["n_params", "cnot_count", "circuit_depth"]:
        if metric in metrics:
            metrics[metric].extend(
                metrics[metric][-1]
                for _ in range(len(metrics["energy"]) - len(metrics[metric]))
            )

    return metrics


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

    s = re.compile("tups", re.IGNORECASE).sub("tUPS", s)
    s = re.compile("ucc", re.IGNORECASE).sub("UCC", s)
    s = re.compile("qeb", re.IGNORECASE).sub("QEB", s)
    s = re.compile("fsd", re.IGNORECASE).sub("FSD", s)
    s = re.compile("gsd", re.IGNORECASE).sub("GSD", s)

    return s


def compare_runs(
    *,
    x_parameter: str | None = None,
    y_parameter: str,
    title: str | None = None,
    x_axis_title: str | None = None,
    y_axis_title: str | None = None,
    group_by: str,
    filter_fixed: dict[str, Any] = {},
    filter_ignored: dict[str, list[Any]] = {},
):
    """
    Comparing multiple runs grouped by a specified parameter, fixed by a specific filter, and with specific x and y axis.

    Args:
        x_parameter (str | None): The parameter for the x axis. Defaults to None.
        y_parameter (str): The parameter to actually plot on the graph, like energy_percent_log.
        title: (str | None): The title for the graph. Defaults to None.
        x_axis_title: (str | None): The x axis title. Defaults to None.
        y_axis_title: (str | None): The y axis title. Defaults to None.
        group_by (str): Parameter name to group runs by (e.g., "optimizer"). Dependent Variable.
        filter_fixed (dict[str, Any]): Dictionary of fixed parameters to filter by. The constant stuff. Defaults to {}.
        filter_ignored (dict[str, list[Any]]): Dictionary of parameters that if satisifed should be ignored. Useful if you want
        to remove some runs from a plot while keeping others. Defaults to {},

    Returns:
        Matplotlib plot.
    """
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])

    grouped_runs = {}  # type: ignore

    for run in runs:
        run_id = run.info.run_id
        params = run.data.params

        for key in params:
            # convert json compatible params into json
            try:
                params[key] = json.loads(params[key])
            except ValueError:
                pass

        if "pool" not in params:
            params["pool"] = {"name": params["starting_ansatz"][1]}

        # Filter for fixed values
        skip = False
        for key, val in filter_fixed.items():
            if get_nested_json(params, key) != val:
                skip = True
                break
        for key, val in filter_ignored.items():
            if get_nested_json(params, key) in val:
                skip = True
                break

        if skip:
            continue

        # Group by selected parameter
        group_val = get_nested_json(params, group_by)

        if group_val is None:
            continue

        grouped_runs.setdefault(group_val, []).append(run_id)

    fig, ax = plt.subplots(figsize=(19.20, 10.80))

    for group, run_ids in grouped_runs.items():
        for run_id in run_ids:
            metrics = get_logged_metrics(run_id)
            if y_parameter not in metrics:
                continue

            x_values, y_values = zip(*metrics[y_parameter])

            if x_parameter is not None:
                if x_parameter not in metrics:
                    continue

                _, x_values = zip(*metrics[x_parameter])

            plt.plot(
                x_values,
                y_values,
                marker="o",
                label=adjust_capitalization(group),
            )

            if (
                y_parameter == "energy_percent_log"
            ):  # plot error bar for each adapt iteration
                adapt_it_x_values = []
                adapt_it_y_values = []

                for i, (x_val, y_val, n_param) in enumerate(
                    zip(x_values, y_values, metrics["n_params"])
                ):
                    if i == 0 or n_param != metrics["n_params"][i - 1]:
                        adapt_it_x_values.append(x_val)
                        adapt_it_y_values.append(y_val)

                plt.errorbar(
                    adapt_it_x_values,
                    adapt_it_y_values,
                    yerr=0.5,
                    fmt="o",
                    color="black",
                )

    formatted_x_parameter = (
        adjust_capitalization(x_parameter) if x_parameter is not None else "Iterations"
    )
    formatted_y_pararmeter = adjust_capitalization(y_parameter)
    formatted_group = adjust_capitalization(group_by.split(".")[0])

    plt.title(
        title
        or f"{formatted_y_pararmeter} vs {formatted_x_parameter} (Grouped by {formatted_group})",
        fontsize=24,
    )
    plt.xlabel(x_axis_title or formatted_x_parameter, fontsize=18)
    plt.ylabel(y_axis_title or formatted_y_pararmeter, fontsize=18)
    plt.legend()
    plt.tight_layout()

    if y_parameter == "energy_percent_log":  # plot chemical accuracy
        plt.axhline(y=log10(0.00159), color="gray", linestyle="--")
        plt.axhspan(ax.get_ylim()[0], log10(0.00159), color="gray", alpha=0.25)

    return fig


def main() -> None:
    molecules = [
        "H2_sto-3g_singlet_H2",
        "H2_6-31g_singlet_H2",
        "H1-Li1_sto-3g_singlet_LiH",
    ]
    optimizers = ["cobyla_optimizer", "lbfgs_optimizer", "trust_region_optimizer"]
    pools = [
        "ucc_ansatz",
        "tups_ansatz",
        "fsd_pool",
        "gsd_pool",
        "qeb_pool",
        "individual_tups_pool",
        "adjacent_tups_pool",
        "multi_tups_pool",
        "full_tups_pool",
    ]
    backends = ["exact", "noisy"]
    observables = ["number_observable", "spin_z_observable", "spin_squared_observable"]
    metrics = ["n_params", "circuit_depth", "cnot_count"]

    for molecule in molecules:
        molecule_name = molecule.split("_")[-1]
        molecule_basis = molecule.split("_")[1]

        # graphs for pools
        for optimizer in optimizers:
            fig = compare_runs(
                y_parameter="energy_percent_log",
                title=f"Energy Error with {adjust_capitalization(optimizer)} on {molecule_name} ({molecule_basis})",
                x_axis_title="VQE Iteration",
                y_axis_title="Energy Error in a.u (Log Scale)",
                group_by="pool.name",
                filter_fixed={
                    "optimizer.name": optimizer,
                    "qiskit_backend.shots": 0,
                    "molecule": molecule,
                },
            )

            Path(f"{OUT_DIR}/pools/{molecule}").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{OUT_DIR}/pools/{molecule}/{optimizer}.png")
            plt.close(fig)

        # for each other metric
        for metric in metrics:
            fig = compare_runs(
                y_parameter=metric,
                title=f"{adjust_capitalization(metric)} with Cobyla on {molecule_name} ({molecule_basis})",
                x_axis_title="VQE Iteration",
                y_axis_title=adjust_capitalization(metric),
                group_by="pool.name",
                filter_fixed={
                    "optimizer.name": "cobyla_optimizer",
                    "qiskit_backend.shots": 0,
                    "molecule": molecule,
                },
            )

            Path(f"{OUT_DIR}/pools/{molecule}").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{OUT_DIR}/pools/{molecule}/{metric}.png")
            plt.close(fig)

        # graphs for optimizers
        for pool, backend in product(pools, backends):
            fig = compare_runs(
                y_parameter="energy_percent_log",
                title=f"Energy Error with {adjust_capitalization(pool)} on {molecule_name} ({molecule_basis}) ({backend})",
                x_axis_title="VQE Iteration",
                y_axis_title="Energy Error in a.u (Log Scale)",
                group_by="optimizer.name",
                filter_fixed={
                    "pool.name": pool,
                    "qiskit_backend.shots": 0 if backend == "exact" else 2**20,
                    "molecule": molecule,
                },
                filter_ignored={
                    "optimizer.name": (
                        ["adam_optimizer", "sgd_optimizer"] if "tups" in pool else []
                    )
                },
            )

            Path(f"{OUT_DIR}/optimizers/{molecule}/{backend}").mkdir(
                parents=True, exist_ok=True
            )
            fig.savefig(f"{OUT_DIR}/optimizers/{molecule}/{backend}/{pool}.png")
            plt.close(fig)

        # graphs for observables
        for observable in observables:
            fig = compare_runs(
                y_parameter=observable,
                title=f"{adjust_capitalization(observable)} with Cobyla and FSD on {molecule_name} ({molecule_basis})",
                x_axis_title="VQE Iteration",
                y_axis_title="Expectation value",
                group_by="optimizer.name",
                filter_fixed={
                    "optimizer.name": "cobyla_optimizer",
                    "pool.name": "fsd_pool",
                    "qiskit_backend.shots": 0,
                    "molecule": molecule,
                },
            )

            Path(f"{OUT_DIR}/observables/{molecule}").mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{OUT_DIR}/observables/{molecule}/{observable}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
