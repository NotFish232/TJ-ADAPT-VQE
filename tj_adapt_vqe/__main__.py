from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
    exact_expectation_value,
)
from .optimizers import LBFGS
from .pools import FullTUPSPool
from .utils import Molecule, make_molecule
from .vqe import ADAPTVQE, ADAPTConvergenceCriteria


def main() -> None:
    mol = make_molecule(Molecule.H2, r=1.5)

    optimizer = LBFGS(lr=0.01, grad_conv_threshold=0.01)

    n_qubits = mol.n_qubits

    observables: list[Observable] = [
        NumberObservable(n_qubits),
        SpinZObservable(n_qubits),
        SpinSquaredObservable(n_qubits),
    ]

    tups = FullTUPSPool(mol)

    vqe = ADAPTVQE(
        mol,
        tups,
        optimizer,
        observables,
        adapt_conv_criteria=ADAPTConvergenceCriteria.LackOfImprovement,
        conv_threshold=1e-3,
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
