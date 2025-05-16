from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
    exact_expectation_value,
)
from .optimizers import Cobyla
from .pools import FullTUPSPool
from .utils import Molecule, make_molecule
from .vqe import VQE, ADAPTConvergenceCriteria
from .utils.ansatz import PerfectPairAnsatz, TUPSAnsatz


def main() -> None:
    mol = make_molecule(Molecule.H5, r=1.5)

    optimizer = Cobyla()

    n_qubits = mol.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    tups = FullTUPSPool(mol)

    vqe = VQE(
        mol,
        optimizer,
        [PerfectPairAnsatz(), TUPSAnsatz(2)],
        observables,
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
