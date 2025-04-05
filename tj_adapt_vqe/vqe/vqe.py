import numpy as np
from openfermion import MolecularData, get_sparse_operator, jordan_wigner
from qiskit.circuit import QuantumCircuit  # type: ignore
from typing_extensions import Self

from ..optimizers import Optimizer
from ..utils import (
    Measure,
    exact_expectation_value,
    make_tups_ansatz,
    openfermion_to_qiskit,
)


class VQE:
    """
    Class implementing the variational quantum eigensolver (VQE) algorithm

    Args:
        molecule: Moleculardata, molecule to find ground state of
        optimizer: Optimizer, optimizer that the Measure class is passed into
        num_shots: int, num shots to run each simulation with

    """

    def __init__(
        self: Self, molecule: MolecularData, optimizer: Optimizer, num_shots: int = 1024
    ) -> None:

        self.molecule = molecule
        self.n_qubits = self.molecule.n_qubits

        molecular_hamiltonian = molecule.get_molecular_hamiltonian()

        self.molecular_hamiltonian_sparse = get_sparse_operator(molecular_hamiltonian)
        self.molecular_hamiltonian_qiskit = openfermion_to_qiskit(
            jordan_wigner(molecular_hamiltonian), molecule.n_qubits
        )

        self.optimizer = optimizer

        self.num_shots = num_shots

        self.circuit = self._make_initial_circuit()

        self.param_vals = (
            np.random.rand(len(self.circuit.parameters)).astype(np.float32) - 0.5
        )

    def _make_initial_circuit(self: Self) -> QuantumCircuit:
        """
        Constructs the parameterized Ansatz circuit to be optimized
        """

        return make_tups_ansatz(self.n_qubits, 1).decompose(reps=2)

    def optimize_parameters(self: Self) -> None:
        """
        Performs a single iteration step of the vqe, stopping when the provided Optimizer's stopping condition has been reached
        """

        # TODO: add logging

        iteration = 1

        while True:
            measure = Measure(
                self.circuit,
                self.param_vals,
                self.molecular_hamiltonian_qiskit,
                num_shots=self.num_shots,
            )

            print(
                f"Iteration: {iteration} | energy={measure.expectation_value:.5f}, param_vals={self.param_vals}, grad={measure.gradients}"
            )

            iteration += 1

            self.param_vals = self.optimizer.update(self.param_vals, measure)

            if self.optimizer.is_converged(measure):
                break

        full_circuit = self.circuit.assign_parameters(
            {p: val for p, val in zip(self.circuit.parameters, self.param_vals)}
        )

        state_ev = exact_expectation_value(
            full_circuit, self.molecular_hamiltonian_sparse
        )

        print(
            f"HF energy: {self.molecule.hf_energy},",
            f"Exact energy: {self.molecule.fci_energy},",
            f"Calculated energy: {state_ev},",
            f"Accuracy: {state_ev / self.molecule.fci_energy:.5%}",
        )
