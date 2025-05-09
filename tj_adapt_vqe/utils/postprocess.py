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


if __name__ == "__main__":
    plot_postprocessing()
