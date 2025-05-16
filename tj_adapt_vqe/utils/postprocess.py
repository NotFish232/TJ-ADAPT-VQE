import json
from functools import reduce

import matplotlib.pyplot as plt
import mlflow
from typing_extensions import Any

RUN_DIR = "./runs/"
mlflow.set_tracking_uri(RUN_DIR)

OUT_DIR = "./results/"


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

    # print(f"found metrics: {list(data.metrics.keys())}")
    metrics = {}

    for k in data.metrics:
        # print(f"\nfetching metric history for: {k}")
        history = client.get_metric_history(run_id, k)
        sorted_history = sorted(
            [(h.step, h.value) for h in history], key=lambda x: x[0]
        )
        metrics[k] = sorted_history

    # some metrics have to be processed

    # make n params same length as energy
    if "n_params" in metrics:
        metrics["n_params"].extend(
            metrics["n_params"][-1]
            for _ in range(len(metrics["energy"]) - len(metrics["n_params"]))
        )

    return metrics


def compare_runs(
    *,
    x_parameter: str | None = None,
    y_parameter: str,
    group_by: str,
    filter_fixed: dict[str, Any] = {},
):
    """
    Comparing multiple runs grouped by a specified parameter, fixed by a specific filter, and with specific x and y axis.

    Args:
        x_parameter (str): The parameter for the x axis
        y_parameter (str): The parameter to actually plot on the graph, like energy_percent_log.
        group_by (str): Parameter name to group runs by (e.g., "optimizer"). Dependent Variable.
        filter_fixed (dict[str, Any]): Dictionary of fixed parameters to filter by. The constant stuff.

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

        if skip:
            continue

        # Group by selected parameter
        group_val = get_nested_json(params, group_by)

        if group_val is None:
            continue

        grouped_runs.setdefault(group_val, []).append(run_id)

    fig = plt.figure(figsize=(13.66, 7.68))

    for group, run_ids in grouped_runs.items():
        for run_id in run_ids:
            metrics = get_logged_metrics(run_id)
            if y_parameter not in metrics:
                continue

            steps, values = zip(*metrics[y_parameter])

            if x_parameter is not None:
                if x_parameter not in metrics:
                    continue

                _, steps = zip(*metrics[x_parameter])

            steps = steps[:200]
            values = values[:200]

            plt.plot(
                steps,
                values,
                marker="o",
                label=" ".join(g.capitalize() for g in group.split("_")),
            )

    formatted_x_parameter = (
        " ".join(m.capitalize() for m in x_parameter.split("_"))
        if x_parameter is not None
        else "Iterations"
    )
    formatted_y_pararmeter = " ".join(m.capitalize() for m in y_parameter.split("_"))
    formatted_group = group_by.split(".")[0].capitalize()

    plt.title(
        f"{formatted_y_pararmeter} vs {formatted_x_parameter} (Grouped by {formatted_group})"
    )
    plt.xlabel(formatted_x_parameter)
    plt.ylabel(formatted_y_pararmeter)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig


def main() -> None:
    for attr in {
        "energy",
        "energy_percent",
        "energy_percent_log",
        "n_params",
        "cnot_count",
        "circuit_depth",
    }:
        fig = compare_runs(
            y_parameter=attr,
            group_by="pool.name",
            filter_fixed={
                "optimizer.name": "cobyla_optimizer",
                "qiskit_backend.shots": 0,
                "molecule": "H2_6-31g_singlet_H2",
            },
        )
        fig.savefig(f"{OUT_DIR}/pool_{attr}.png")


if __name__ == "__main__":
    main()
