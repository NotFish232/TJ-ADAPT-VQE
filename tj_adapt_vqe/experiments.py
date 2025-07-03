from itertools import product
from multiprocessing import Pool as MPPool

from openfermion import MolecularData

from .observables import (
    EXACT_BACKEND,
    SHOT_NOISE_BACKEND,
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .optimizers import LBFGS, SGD, Adam, Cobyla, Optimizer, TrustRegion
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
from .utils import (
    Ansatz,
    HartreeFockAnsatz,
    Molecule,
    QiskitUCCSDAnsatz,
    TUPSAnsatz,
    UCCAnsatz,
    make_molecule,
)
from .vqe import ADAPTVQE, VQE, ADAPTConvergenceCriteria

NUM_PROCESSES = 16


def make_molecule_from_str(molecule_str: str, r: float) -> MolecularData:
    return make_molecule(Molecule[molecule_str], r)


def make_pool_from_str(pool_str: str, molecule: MolecularData) -> Pool:
    if pool_str == "AdjacentTUPS":
        return AdjacentTUPSPool(molecule)
    if pool_str == "FSD":
        return FSDPool(molecule, 2)
    if pool_str == "FullTUPS":
        return FullTUPSPool(molecule)
    if pool_str == "GSD":
        return GSDPool(molecule, 2)
    if pool_str == "IndividualTUPS":
        return IndividualTUPSPool(molecule)
    if pool_str == "MultiTUPS":
        return MultiTUPSPool(molecule)
    if pool_str == "UnrestrictedTUPS":
        return UnrestrictedTUPSPool(molecule)
    if pool_str == "UnresIndividualTUPS":
        return UnresIndividualTUPSPool(molecule)
    if pool_str == "QEB":
        return QEBPool(molecule, 2)
    if pool_str == "StandardTUPS":
        return None  # type: ignore
    if pool_str == "UCCSD":
        return None  # type: ignore
    if pool_str == "QiskitUCCSD":
        return None  # type: ignore

    raise NotImplementedError()


def make_optimizer_from_str(optimizer_str: str) -> Optimizer:
    if optimizer_str == "Adam":
        return Adam(0.01)
    if optimizer_str == "Cobyla":
        return Cobyla()
    if optimizer_str == "LBFGS":
        return LBFGS()
    if optimizer_str == "SGD":
        return SGD(0.01)
    if optimizer_str == "TrustRegion":
        return TrustRegion()

    raise NotImplementedError()


def train_function(params: tuple[str, str, str, str]) -> None:
    pool_str, optimizer_str, qiskit_backend_str, molecule_str = params

    molecule = make_molecule_from_str(molecule_str, 1.5)
    pool = make_pool_from_str(pool_str, molecule)
    optimizer = make_optimizer_from_str(optimizer_str)

    if qiskit_backend_str == "exact":
        qiskit_backend = EXACT_BACKEND
    if qiskit_backend_str == "noisy":
        qiskit_backend = SHOT_NOISE_BACKEND

    starting_ansatz: list[Ansatz] = [HartreeFockAnsatz()]

    # use perfect pair configuration for all TUPS ansatz
    if isinstance(
        pool,
        (
            AdjacentTUPSPool,
            FullTUPSPool,
            IndividualTUPSPool,
            MultiTUPSPool,
            UnrestrictedTUPSPool,
        ),
    ):
        # starting_ansatz = [PerfectPairAnsatz()]
        starting_ansatz = [HartreeFockAnsatz()]

    # custom starting ansatz for non pool pools
    if pool_str == "StandardTUPS":
        n_layers = 3
        if molecule_str == "LiH":
            n_layers = 5
        if molecule_str == "BeH2":
            n_layers = 7
        if molecule_str == "H6":
            n_layers = 9

        # starting_ansatz = [PerfectPairAnsatz(), TUPSAnsatz(n_layers)]
        starting_ansatz = [HartreeFockAnsatz(), TUPSAnsatz(n_layers)]

    if pool_str == "UCCSD":
        starting_ansatz = [HartreeFockAnsatz(), UCCAnsatz(2)]
    elif pool_str == "QiskitUCCSD":
        starting_ansatz = [HartreeFockAnsatz(), QiskitUCCSDAnsatz()]

    n_qubits = molecule.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    if pool is not None:
        criteria = ADAPTConvergenceCriteria.Gradient
        conv_threshold = 2e-2

        # larger thresholds for noisy backends
        if qiskit_backend_str == "noisy":
            conv_threshold = 5e-2

        # These types of pools should use the LackOfImprovement criteria
        if len(pool) == 1:
            criteria = ADAPTConvergenceCriteria.LackOfImprovement
            conv_threshold = 2e-4

            if qiskit_backend_str == "noisy":
                conv_threshold = 1e-3

        # maximum number of adapt iterations, scale by molecule
        max_adapt_iter = 10
        if molecule_str == "LiH":
            max_adapt_iter = 20
        if molecule_str == "BeH2":
            max_adapt_iter = 30
        if molecule_str == "H6":
            max_adapt_iter = 50

        vqe = ADAPTVQE(
            molecule,
            pool,
            optimizer,
            starting_ansatz,
            observables,
            max_adapt_iter=max_adapt_iter,
            qiskit_backend=qiskit_backend,
            adapt_conv_criteria=criteria,
            conv_threshold=conv_threshold,
        )
    else:
        vqe = VQE(
            molecule,
            optimizer,
            starting_ansatz,
            observables,
            qiskit_backend=qiskit_backend,
        )

    vqe.run(False)


def main() -> None:
    molecules = ["H2", "LiH"]  # , "BeH2", "H6"]

    qiskit_backends = ["exact", "noisy"]

    optimizers = ["Cobyla", "LBFGS", "TrustRegion", "Adam", "SGD"]
    pools = [
        "FSD",
        "GSD",
        "QEB",
        "FullTUPS",
        "IndividualTUPS",
        "AdjacentTUPS",
        "MultiTUPS",
        "UnrestrictedTUPS",
        "StandardTUPS",
        "UCCSD",
        "QiskitUCCSD",
    ]

    # do this loop seperate because drastically different compute times
    for molecule in molecules:
        with MPPool(NUM_PROCESSES) as p:
            p.map(
                train_function, product(pools, optimizers, qiskit_backends, [molecule])
            )


if __name__ == "__main__":
    main()
