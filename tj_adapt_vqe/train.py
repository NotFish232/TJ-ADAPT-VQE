from .ansatz import HartreeFockAnsatz
from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .observables.measure import exact_expectation_value
from .optimizers import LBFGSOptimizer
from .pools import UnrestrictedTUPSPool
from .utils.molecules import Molecule
from .vqe import ADAPTVQE


def main() -> None:
    mol = Molecule.H4(1.5)

    pool = UnrestrictedTUPSPool(mol)

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
        conv_threshold=1e-3,
    )
    vqe.run()

    final_energy = exact_expectation_value(
        vqe.circuit.assign_parameters(
            {p: v for p, v in zip(vqe.circuit.parameters, vqe.param_vals)}
        ),
        vqe.hamiltonian.operator_sparse,
    )
    target_energy = vqe.molecule.data.fci_energy
    print(f"Energy {final_energy} ({abs(final_energy - target_energy):e})")


if __name__ == "__main__":
    main()
