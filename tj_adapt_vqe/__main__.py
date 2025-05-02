from openfermion import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore

from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
    exact_expectation_value,
)
from .optimizers import Adam
from .pools import FSDPool
from .vqe import ADAPTVQE


def main() -> None:
    r = 1.5
    h2 = MolecularData(
        [["H", [0, 0, 0]], ["H", [0, 0, r]]], "sto-3g", 1, 0, description="H2"
    )
    h2 = run_pyscf(h2, run_fci=True, run_ccsd=True)
    # lih = MolecularData([["Li", [0, 0, 0]], ["H", [0, 0, r]]], "sto-3g", 1, 0, "LiH")
    # # lih = run_pyscf(lih, run_fci=True, run_ccsd=True)
    # beh2 = MolecularData([["Be", [0, 0, 0]], ["H", [0, 0, r]], ["H", [0, 0, -r]]], 'sto-3g', 1, 0, 'BeH2')
    # beh2 = run_pyscf(beh2, run_fci=True, run_ccsd=True)
    # h6 = MolecularData([('H', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, 2 * r)),
    #             ('H', (0, 0, 3 * 2 * r)), ('H', (0, 0, 4 * r)), ('H', (0, 0, 5 * r))], 'sto-3g', 1, 0, description='H6')
    # h6 = run_pyscf(h6, run_fci=True, run_ccsd=True)

    mol = h2

    optimizer = Adam(lr=0.01, gradient_convergence_threshold=0.01)

    n_qubits = mol.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    fsd = FSDPool(mol, 2)

    vqe = ADAPTVQE(mol, fsd, optimizer, observables)
    vqe.run()

    final_energy = exact_expectation_value(
        vqe.circuit.assign_parameters(
            {p: v for p, v in zip(vqe.circuit.parameters, vqe.param_vals)}
        ),
        vqe.hamiltonian.operator_sparse,
    )
    target_energy = vqe.molecule.fci_energy
    print(
        f"Energy {final_energy} ({abs(final_energy - target_energy):e})"
    )

  

if __name__ == "__main__":
    main()