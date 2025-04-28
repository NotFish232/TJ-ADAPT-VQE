from openfermion import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore

from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .optimizers import BFGS
from .vqe import VQE


def main() -> None:
    # h2 = MolecularData([["H", [0, 0, 0]], ["H", [0, 0, 1.5]]], "sto-3g", 1, 0, description="H2")
    # h2 = run_pyscf(h2, run_fci=True, run_ccsd=True)
    lih = MolecularData([["Li", [0, 0, 0]], ["H", [0, 0, 1]]], "sto-3g", 1, 0, "LiH")
    lih = run_pyscf(lih)
    # beh2 = MolecularData([["Be", [0, 0, 0]], ["H", [0, 0, 2]], ["H", [0, 0, -2]]], 'sto-3g', 1, 0, 'BeH2')
    # beh2 = run_pyscf(beh2)
    mol = lih

    optimizer = BFGS(learning_rate=0.1)

    n_qubits = mol.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    vqe = VQE(mol, optimizer, observables)
    vqe.optimize_parameters()


if __name__ == "__main__":
    main()
