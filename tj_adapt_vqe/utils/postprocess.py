import json

import matplotlib.pyplot as plt
import mlflow

RUN_DIR = "./runs/"
mlflow.set_tracking_uri(RUN_DIR)


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
        """
        for step, val in sorted_history:
            print(f"step {step}: {val}")
        """
        metrics[k] = sorted_history

    return metrics


def plot_postprocessing(run_name: str = "ADAPTVQE Run"):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"], filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    # wrong id
    if not runs:
        print(f"No runs found with name '{run_name}'")
        return

    run = runs[0]
    run_id = run.info.run_id
    print(f"using run_id: {run_id}")

    metrics = get_logged_metrics(run_id)

    # example plot for sampling noise group
    if "energy_percent" in metrics:
        steps, errors = zip(*metrics["energy_percent"])

        plt.figure()
        plt.plot(steps, errors, marker="o")
        plt.xlabel("Iterations")
        plt.ylabel("Energy")
        plt.title("Energy vs. Iterations")
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig("energy_error_plot.png")
        plt.show()
    else:
        print("metric is not plotted ")


def compare_runs(
    group_by: str, run_name: str = "ADAPTVQE Run", filter_fixed: dict[str, str] = {}
):
    """
    Comparing mulitple runs grouped by a specified parameter.

    Args:
        group_by (str): Parameter name to group runs by (e.g., "optimizer"). Dependent Variable.
        run_name (str): MLflow run name to search for.
        filter_fixed (dict): Dictionary of fixed parameters to filter by. The constant stuff.

    Returns:
        Matplotlib plot.
    """
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"], filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    if not runs:
        print(f"No runs found with name '{run_name}'")
        return

    grouped_runs = {}  # type: ignore

    for run in runs:
        run_id = run.info.run_id
        params = run.data.params

        # Filter for fixed values
        skip = False
        for key, val in filter_fixed.items():
            if key not in params or params[key] != val:
                skip = True
                break
        if skip:
            continue

        # Group by selected parameter
        group_val = params.get(group_by)
        if not group_val:
            continue

        short_label = group_val
        # this part doens't work because group_val is a string which is weird
        # @JUSTIN CHANGE IT json and fix formatting its not pretty rn
        try:
            parsed = json.loads(group_val.replace("'", '"'))
            if isinstance(parsed, dict) and "name" in parsed:
                short_label = parsed["name"]
        except Exception:
            pass

        grouped_runs.setdefault(short_label, []).append(run_id)

    plt.figure()
    for group, run_ids in grouped_runs.items():
        for run_id in run_ids:
            metrics = get_logged_metrics(run_id)
            if "energy" not in metrics:
                continue
            steps, values = zip(*metrics["energy"])
            plt.plot(steps, values, marker="o", label=group)

    plt.title(f"Energy vs Iterations (Grouped by {group_by})")
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # plot_postprocessing()
    compare_runs(
        group_by="optimizer",
        filter_fixed={
            "pool": "{'name': 'full_tups_pool', 'n_qubits': 4}",
            "molecule": "H2_sto-3g_singlet_H2",
        },
    )
