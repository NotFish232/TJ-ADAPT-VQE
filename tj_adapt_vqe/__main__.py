from .observables import (
    EXACT_BACKEND,
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
    exact_expectation_value,
)
from .optimizers import Cobyla
from .pools import (
    FSDPool,
    FullTUPSPool,
    GSDPool,
    IndividualTUPSPool,
    MultiTUPSPool,
    QEBPool,
)
from .utils import HartreeFockAnsatz, Molecule, make_molecule
from .vqe import ADAPTVQE


def main() -> None:
    mol = make_molecule(Molecule.H2, r=1.5)

    optimizer = Cobyla()

    n_qubits = mol.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    pools = [
        FSDPool(mol, 2),
        FSDPool(mol, 2),
        FullTUPSPool(mol),
        GSDPool(mol, 2),
        IndividualTUPSPool(mol),
        MultiTUPSPool(mol),
        QEBPool(mol, 2),
    ]

    for pool in pools:
        vqe = ADAPTVQE(
            mol,
            pool,
            optimizer,
            [HartreeFockAnsatz()],
            observables,
            qiskit_backend=EXACT_BACKEND,
        )
        vqe.run()

        final_energy = exact_expectation_value(
            vqe.circuit.assign_parameters(
                {p: v for p, v in zip(vqe.circuit.parameters, vqe.param_vals)}
            ),
            vqe.hamiltonian.operator_sparse,
        )
        target_energy = vqe.molecule.fci_energy
        print(f"Energy {final_energy} ({abs(final_energy - target_energy):e})")


if __name__ == "__main__":
    main()
