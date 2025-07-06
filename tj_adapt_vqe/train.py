from .ansatz import HartreeFockAnsatz
from .pools import UnresIndividualTUPSPool
from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .observables.measure import exact_expectation_value
from .optimizers import LBFGSOptimizer
from .utils.molecules import Molecule, make_molecule
from .vqe import ADAPTVQE


def main() -> None:
    mol = make_molecule(Molecule.H4, r=1.5)

    pool = UnresIndividualTUPSPool(mol)

    optimizer = LBFGSOptimizer()

    n_qubits = mol.n_qubits

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
        conv_threshold=1e-3
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
