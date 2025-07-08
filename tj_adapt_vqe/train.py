from .ansatz import HartreeFockAnsatz
from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .optimizers import LBFGSOptimizer
from .pools import UnresIndividualTUPSPool
from .utils.molecules import Molecule
from .vqe import ADAPTVQE


def main() -> None:
    mol = Molecule.H4(1.5)

    pool = UnresIndividualTUPSPool(mol)

    optimizer = LBFGSOptimizer()

    n_qubits = mol.data.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    vqe = ADAPTVQE(
        mol,
        pool,
        optimizer,
        [HartreeFockAnsatz()],
        observables,
        max_adapt_iter=-1,
        conv_threshold=1e-4,
    )
    vqe.run()


if __name__ == "__main__":
    main()
