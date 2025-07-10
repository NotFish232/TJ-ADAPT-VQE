from itertools import product
from multiprocessing import Pool as MPPool

from typing_extensions import Any, Type

from .ansatz import Ansatz, HartreeFockAnsatz
from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .observables.qiskit_backend import QiskitBackend
from .optimizers import LBFGSOptimizer, Optimizer
from .pools import (
    FSDPool,
    GSDPool,
    Pool,
    QEBPool,
    UnresIndividualTUPSPool,
)
from .utils.molecules import Molecule
from .vqe import ADAPTVQE, ADAPTConvergenceCriteria

NUM_PROCESSES = 16


def train_function(
    params: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    pool_conf, optimizer_conf, qiskit_backend_conf, molecule_conf = params

    molecule = Molecule.from_config(molecule_conf)
    qiskit_backend = QiskitBackend.from_config(qiskit_backend_conf)
    pool = Pool.from_config(pool_conf, molecule=molecule)
    optimizer = Optimizer.from_config(optimizer_conf)

    n_qubits = molecule.data.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    starting_ansatz: list[Ansatz] = [HartreeFockAnsatz()]

    max_adapt_iter = 4 * n_qubits

    vqe = ADAPTVQE(
        molecule,
        pool,
        optimizer,
        starting_ansatz,
        observables,
        max_adapt_iter=max_adapt_iter,
        adapt_conv_criteria=ADAPTConvergenceCriteria.ErrorPercent,
        conv_threshold=1e-4,
        qiskit_backend=qiskit_backend,
    )

    vqe.run(False)


def main() -> None:
    molecules_t = [
        Molecule.H2,
        Molecule.H2_631G,
        Molecule.H4,
        Molecule.LiH,
        Molecule.H5,
        Molecule.H6,
        Molecule.H4_631G,
    ]
    qiskit_backends_t = [QiskitBackend.Exact]
    optimizers_t = [LBFGSOptimizer]
    pools_t: list[Type[Pool]] = [
        FSDPool,
        GSDPool,
        QEBPool,
        UnresIndividualTUPSPool,
    ]
    r = 1.5

    qiskit_backend_confs = [q().to_config() for q in qiskit_backends_t]
    optimizer_confs = [{"_name": o._name(), "_type": o._type()} for o in optimizers_t]
    pool_confs = [{"_name": p._name(), "_type": p._type()} for p in pools_t]

    # do this loop seperate because drastically different compute times
    for molecule_t in molecules_t:
        molcule_conf = molecule_t(r, run_fci=False).to_config()
        molcule_conf["run_fci"] = True

        args = product(
            pool_confs, optimizer_confs, qiskit_backend_confs, [molcule_conf]
        )

        with MPPool(NUM_PROCESSES) as p:
            p.map(train_function, args)


if __name__ == "__main__":
    main()
