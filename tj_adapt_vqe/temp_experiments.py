from itertools import product
from multiprocessing import Pool as MPPool
from typing_extensions import Any

from .ansatz import Ansatz, HartreeFockAnsatz, QiskitUCCSDAnsatz, TUPSAnsatz, UCCAnsatz
from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .observables.measure import EXACT_BACKEND, SHOT_NOISE_BACKEND
from .optimizers import (
    AdamOptimizer,
    CobylaOptimizer,
    LBFGSOptimizer,
    Optimizer,
    SGDOptimizer,
    TrustRegionOptimizer,
)
from .pools import (
    AdjacentTUPSPool,
    FSDPool,
    FullTUPSPool,
    GSDPool,
    IndividualTUPSPool,
    MultiTUPSPool,
    Pool,
    QEBPool,
    UnresIndividualTUPSPool,
    UnrestrictedTUPSPool,
)
from .utils.molecules import Molecule, MoleculeConstructor
from .vqe import ADAPTVQE, VQE, ADAPTConvergenceCriteria

NUM_PROCESSES = 16


def train_function(
    params: tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]],
) -> None:
    pool_conf, optimizer_conf, qiskit_backend_s, molecule_conf = params

    molecule = Molecule.from_config(molecule_conf)
    pool = Pool.from_config(pool_conf, molecule=molecule)
    optimizer = Optimizer.from_config(optimizer_conf)

    if qiskit_backend_s == "exact":
        qiskit_backend = EXACT_BACKEND
    if qiskit_backend_s == "noisy":
        qiskit_backend = SHOT_NOISE_BACKEND

    starting_ansatz: list[Ansatz] = [HartreeFockAnsatz()]

    n_qubits = molecule.data.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    max_adapt_iter = -1
    if pool is UnrestrictedTUPSPool:
        max_adapt_iter = 4 * n_qubits

    vqe = ADAPTVQE(
        molecule,
        pool,
        optimizer,
        starting_ansatz,
        observables,
        max_adapt_iter=max_adapt_iter,
        qiskit_backend=qiskit_backend,
    )


    vqe.run(False)


def main() -> None:
    r = 1.5

    optimizers = [LBFGSOptimizer()]
    pool_t = [
        FSDPool,
        GSDPool,
        QEBPool,
        UnrestrictedTUPSPool,
        UnresIndividualTUPSPool,
    ]
    molecules = [
        Molecule.H2(r),
        Molecule.H2_631G(r),
        Molecule.H4(r),
        Molecule.LiH(r),
        Molecule.H4_631G(r),
        Molecule.H5(r),
        Molecule.H6(r),
    ]
    qiskit_backends = ["exact", "noisy"]

    # do this loop seperate because drastically different compute times
    for molecule in molecules:
        pools = [p_t(molecule) for p_t in pool_t]

        molecule_config = molecule.to_config()
        pool_configs = [p.to_config() for p in pools]
        optimizer_configs = [o.to_config() for o in optimizers]
        with MPPool(NUM_PROCESSES) as p:
            p.map(
                train_function,
                product(pool_configs, optimizer_configs, qiskit_backends, [molecule_config]),
            )


if __name__ == "__main__":
    main()
