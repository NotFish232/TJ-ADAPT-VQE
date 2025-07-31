import json
import re
import subprocess
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
    ("lbfgs", "LBFGS"),
    ("cnot", "CNOT"),
]

CHEMICAL_ACCURACY = 0.00159


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
    params = get_run_params(run_id)

    client = MlflowClient()
    raw_metrics = client.get_run(run_id).data.metrics

    metrics = {}

    for metric in raw_metrics:
        history = client.get_metric_history(run_id, metric)
        sorted_history = sorted(
            [(h.step, h.value) for h in history], key=lambda x: x[0]
        )
        metrics[metric] = sorted_history

    # manually add adapt energy as a metric, even tho this should ideally be done in the adapt vqe code
    if "adapt_energy_percent" not in metrics:
        metrics["adapt_energy_percent"] = [
            (x, abs(y - params["fci_energy"])) for x, y in metrics["adapt_energy"]
        ]

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


def compile_latex(path_s: str) -> None:
    """
    Tries to compile .tex file at path into the same directory but replacing the .tex file extension
    with .pdf

    Args:
        path_s (str): The path to the latex fiule
    """

    # if pdflatex doesn't exist don't try to compile
    try:
        subprocess.run(["pdflatex", "--version"], stdout=subprocess.DEVNULL)
    except Exception:
        return

    path = Path(path_s)

    dir = path.parent
    filename = path.stem

    with open(path, "r") as f:
        latex = f.read()

    wrapped_latex = (
        "\\documentclass{article}\n"
        "\\usepackage{tikz}\n"
        "\\usepackage{pgfplots}\n"
        "\\usepackage{float}\n"
        "\\begin{document}\n"
        "\\thispagestyle{empty}\n"
        "\\begin{center}\n"
        f"{latex}"
        "\\end{center}\n"
        "\\end{document}\n"
    )

    with open(f"{dir}/temp.tex", "w") as f:
        f.write(wrapped_latex)

    subprocess.run(
        [
            "pdflatex",
            f"-output-directory={dir}",
            f"-jobname={filename}",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"{dir}/temp.tex",
        ],
        stdout=subprocess.DEVNULL,
    )

    Path(f"{dir}/temp.tex").unlink()
    Path(f"{dir}/{filename}.aux").unlink(missing_ok=True)
    Path(f"{dir}/{filename}.log").unlink(missing_ok=True)


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
) -> tuple[Figure, str] | None:
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

    if len(grouped_runs) == 0:
        return None

    sorted_runs = sorted(grouped_runs.items())

    fig, ax = plt.subplots(constrained_layout=True, figsize=(19.20, 10.80))

    # technically you want a min of the energy and a max of the iter but it doesn't matter since floats are never equal
    if grouped_runs and truncate_runs:
        max_iter = min(
            get_run_metrics(run_id)[y_parameter][-1][::-1]
            for run_ids in grouped_runs.values()
            for run_id in run_ids
        )[1]

    # maps run_id to (x_vals, y_vals)
    graph_coordinates = {}

    # maps run_id to tuple of (x, y, color)
    error_bars: dict[str, list] = {}

    for group, run_ids in sorted_runs:
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

            graph_coordinates[run_id] = (x_vals, y_vals)

            l = plt.plot(x_vals, y_vals, marker="o", label=adjust_capitalization(group))

            if adapt_bars and x_parameter is None:
                error_bars[run_id] = []

                color = l[0].get_color()

                # plot error bar for each adapt iteration if x parameter is just time
                for i, (t, n_param) in enumerate(metrics["n_params"]):
                    if truncate_runs and t - 1 >= max_iter + 5:
                        break

                    if i == 0 or n_param != metrics["n_params"][i - 1]:
                        x_i = x_vals[t - 1]
                        y_i = y_vals[t - 1]

                    error_bars[run_id].append((x_i, y_i, color))

    if y_parameter.endswith("energy_percent"):  # plot chemical accuracy
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
    x0, y0, x1, y1 = ax.viewLim.bounds
    for x_i, y_i, color in (tup for v in error_bars.values() for tup in v):
        new_color = "#" + "".join(
            hex(min(15, (ord(c) - 48) % 39 + 4))[2] for c in color[1:]  # type: ignore
        )

        if log_scale:
            y_norm = abs(log(y_i / abs(y0), y1 / y0))
        else:
            y_norm = (y_i - y0) / (y1 - y0)

        plt.axvline(x_i, ymin=y_norm - 0.05, ymax=y_norm + 0.05, color=new_color)

    s = (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        f"   axis on top,\n"
        f"   title={{{title}}},\n"
        f"   xlabel={{{x_axis_title}}},\n"
        f"   ylabel={{{y_axis_title}}},\n"
        f"   xmin={x0}, xmax={x0 + x1},\n"
        f"   ymin={y0}, ymax={y0 + y1},\n"
        "   legend pos=south west,\n"
        "   legend style={font={\\tiny}, fill opacity=0.9},\n"
        "   width=8cm,\n"
        "   height=5cm,\n"
    )
    if log_scale:
        s += (
            "   ymode=log,\n"
            "   minor y tick style={draw=none},\n"
        )
    s += "]\n\n"

    if y_parameter.endswith("energy_percent"):
        s += (
            "\\addplot [draw=none, fill=gray!20, forget plot]\n"
            f"coordinates {{({x0},{y0}) ({x0 + x1},{y0}) ({x0 + x1},{CHEMICAL_ACCURACY}) ({x0},{CHEMICAL_ACCURACY})}}\n"
            "-- cycle;\n"
            f"\\addplot [domain={x0}:{x0 + x1}, samples=2, dotted, thick, color=gray] {{{CHEMICAL_ACCURACY}}};\n"
            f"\\addlegendentry{{Chemical Accuracy ({CHEMICAL_ACCURACY} $E_h$)}}\n"
        )

    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "violet",
        "orange",
        "lime",
        "teal",
    ]
    for color, (group, run_ids) in zip(colors, sorted_runs):
        for run_id in run_ids:
            x_vals, y_vals = graph_coordinates[run_id]

            s += (
                "\\addplot[\n"
                f"  color={color},\n"
                "   mark=*,\n"
                "   mark size=1.5pt,\n"
                "] coordinates {\n"
                f"   {' '.join(f'({x},{y})' for x, y in zip(x_vals, y_vals))}\n"
                "};\n"
                f"\\addlegendentry{{{adjust_capitalization(group)}}}\n"
            )

            if err := error_bars.get(run_id):
                s += (
                    "\\addplot[\n"
                    "   only marks,\n"
                    f"  color={color}!80,\n"
                    "   mark=|,\n"
                    "   mark size=15pt,\n"
                    "   forget plot,\n"
                    "] coordinates {\n"
                    f"   {' '.join(f'({x},{y})' for x, y, _ in err)}\n"
                    "};\n"
                )

    s += (
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\caption{}\n"
        "\\label{fig:fig_label}\n"
        "\\end{figure}\n"
    )

    return fig, s


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
            if res := compare_runs(
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
            ):
                fig, latex = res

                pools_dir = f"{molecule_dir}/pools/{optimizer._name()}"
                Path(pools_dir).mkdir(parents=True, exist_ok=True)

                fig.savefig(f"{pools_dir}/graph.png")
                plt.close(fig)

                with open(f"{pools_dir}/latex.tex", "w") as f:
                    f.write(latex)
                compile_latex(f"{pools_dir}/latex.tex")

            if res := compare_runs(
                group_by="pool._name",
                y_parameter="adapt_energy_percent",
                title=f"Energy Error with {adjust_capitalization(optimizer._name())} on {molecule.name} ({molecule.basis})",
                x_axis_title="ADAPT VQE Iterations",
                y_axis_title="Energy Error in a.u",
                filter_fixed={
                    "optimizer._name": optimizer._name(),
                    "qiskit_backend.shots": 0,
                    "molecule.name": molecule.name,
                    "molecule.basis": molecule.basis,
                },
                log_scale=True,
                truncate_runs=True,
            ):
                fig, latex = res

                pools_dir = f"{molecule_dir}/pools/adapt_{optimizer._name()}"
                Path(pools_dir).mkdir(parents=True, exist_ok=True)

                fig.savefig(f"{pools_dir}/graph.png")
                plt.close(fig)

                with open(f"{pools_dir}/latex.tex", "w") as f:
                    f.write(latex)
                compile_latex(f"{pools_dir}/latex.tex")

        # for each other metric
        for metric in metrics:
            if res := compare_runs(
                group_by="pool._name",
                y_parameter=metric,
                title=f"{adjust_capitalization(metric)} with LBFGS on {molecule.name} ({molecule.basis})",
                x_axis_title="Cumulative VQE Iterations",
                y_axis_title=adjust_capitalization(metric),
                filter_fixed={
                    "optimizer._name": LBFGSOptimizer._name(),
                    "qiskit_backend.shots": 0,
                    "molecule.name": molecule.name,
                    "molecule.basis": molecule.basis,
                },
                truncate_runs=True,
            ):
                fig, latex = res

                metric_dir = f"{molecule_dir}/pools/{metric}"
                Path(metric_dir).mkdir(parents=True, exist_ok=True)

                fig.savefig(f"{metric_dir}/graph.png")
                plt.close(fig)

                with open(f"{metric_dir}/latex.tex", "w") as f:
                    f.write(latex)
                compile_latex(f"{metric_dir}/latex.tex")

            # other special graphs for each metric with them as the x parameter
            if res := compare_runs(
                group_by="pool._name",
                x_parameter=metric,
                y_parameter="energy_percent",
                title=f"Energy Error by {adjust_capitalization(metric)} with LBFGS on {molecule.name} ({molecule.basis})",
                x_axis_title=adjust_capitalization(metric),
                y_axis_title="Energy Error in a.u",
                filter_fixed={
                    "optimizer._name": LBFGSOptimizer._name(),
                    "qiskit_backend.shots": 0,
                    "molecule.name": molecule.name,
                    "molecule.basis": molecule.basis,
                },
                log_scale=True,
                truncate_runs=True,
            ):
                fig, latex = res

                metric_dir = f"{molecule_dir}/pools/{metric}_energy"
                Path(metric_dir).mkdir(parents=True, exist_ok=True)

                fig.savefig(f"{metric_dir}/graph.png")
                plt.close(fig)

                with open(f"{metric_dir}/latex.tex", "w") as f:
                    f.write(latex)
                compile_latex(f"{metric_dir}/latex.tex")

        # graphs for observables
        for observable in observables:
            if res := compare_runs(
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
            ):
                fig, latex = res

                observable_dir = f"{molecule_dir}/observables/{observable._name()}"
                Path(observable_dir).mkdir(parents=True, exist_ok=True)

                fig.savefig(f"{observable_dir}/graph.png")
                plt.close(fig)

                with open(f"{observable_dir}/latex.tex", "w") as f:
                    f.write(latex)
                compile_latex(f"{observable_dir}/latex.tex")


if __name__ == "__main__":
    main()
