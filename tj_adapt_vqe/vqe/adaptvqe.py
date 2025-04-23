import numpy as np
from openfermion import MolecularData
from qiskit.circuit import Parameter  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore
from typing_extensions import Self

from ..observables import Measure, Observable, SparsePauliObservable, exact_expectation_value
from ..optimizers import Optimizer
from ..pools import Pool
from .vqe import VQE


class ADAPTVQE(VQE):
    """
    Class implementing the ADAPT-VQE algorithm
    """

    def __init__(
        self: Self,
        molecule: MolecularData,
        pool: Pool,
        optimizer: Optimizer,
        observables: list[Observable],
        num_shots: int = 1024,
    ) -> None:
        """
        Initializes the ADAPTVQE object
        Args:
            molecule (MolecularData): the molecular data that is used for the adapt vqe
            pool (Pool): the pool that the ADAPTVQE uses to form the Ansatz
            Other arguments are passed directly to the VQE constructor
        """

        super().__init__(molecule, optimizer, observables, num_shots)

        self.pool = pool

        self.commutators = self._calculate_commutators()

        self.logger.add_config_option("pool", self.pool.to_config())

    def _calculate_commutators(self: Self) -> list[Observable]:
        """
        Calculates the commutator between the Hamiltonian and every Observable
        within the provided Pool
        """

        H = self.hamiltonian.operator

        return [
            SparsePauliObservable(
                (1j * (H @ A - A @ H).simplify()).simplify(),
                f"commutator_{i}",
                self.n_qubits,
            )
            for i, A in enumerate(self.pool.operators)
        ]

    def _find_best_operator(self: Self) -> tuple[float, int]:
        """
        Returns the gradient and idx of the best operator  within the pool that maximizes the commutator with the hamiltonian

        """

        m = Measure(
            self.circuit, self.param_vals, self.commutators, num_shots=self.num_shots
        )

        grads = np.abs([m.evs[c] for c in self.commutators])

        idx = np.argmax(grads).item()

        return grads[idx], idx

    def run(self: Self) -> None:
        """
        Runs the ADAPT-VQE Algorithm
        """

        iteration = 1

        while True:
            max_grad, max_idx = self._find_best_operator()

            self.logger.add_logged_value("new_operator", max_idx)
            self.logger.add_logged_value("new_operator_grad", max_grad)

            if abs(max_grad) < 0.04:  # chosen by trial and error
                break

            self.param_vals = np.append(
                self.param_vals, np.random.rand(1).astype(np.float32) - 0.5
            )
            param = Parameter(f"n{iteration}{self.pool.labels[max_idx]}")
            self.circuit.compose(
                PauliEvolutionGate(self.pool.operators[max_idx], param), inplace=True
            )
            self.circuit = self.circuit.decompose(reps=2)

            self.optimize_parameters()

            iteration += 1

        full_circuit = self.circuit.assign_parameters(
            {p: val for p, val in zip(self.circuit.parameters, self.param_vals)}
        )

        state_ev = exact_expectation_value(
            full_circuit, self.hamiltonian.operator_sparse
        )

        print(
            f"ADAPT VQE[HF energy: {self.molecule.hf_energy},",
            f"Exact energy: {self.molecule.fci_energy},",
            f"Calculated energy: {state_ev},",
            f"Accuracy: {state_ev / self.molecule.fci_energy:.5%}]",
        )
