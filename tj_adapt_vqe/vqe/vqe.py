import numpy as np
from openfermion import MolecularData
from qiskit.circuit import QuantumCircuit  # type: ignore
from typing_extensions import Self

from ..observables.observable import HamiltonianObservable
from ..optimizers.optimizer import Optimizer
from ..utils.ansatz import make_perfect_pair_ansatz, make_tups_ansatz
from ..utils.measure import (
    Measure,
    exact_expectation_value,
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
        self.hamiltonian = HamiltonianObservable(molecule)
        self.n_qubits = self.molecule.n_qubits

        self.optimizer = optimizer

        self.num_shots = num_shots

        self.circuit = self._make_ansatz()

        self.param_vals = 2 * np.random.rand(len(self.circuit.parameters)) - 1



    def _make_ansatz(self: Self) -> QuantumCircuit:
        ansatz = make_perfect_pair_ansatz(self.n_qubits).compose(make_tups_ansatz(self.n_qubits, 1))
   
        return ansatz.decompose(reps=2)

    def optimize_parameters(self: Self, should_print: bool = True) -> None:
        """
        Performs a single iteration step of the vqe, stopping when the provided Optimizer's stopping condition has been reached
        """

        # TODO: add logging

        iteration = 1

        while True:
            measure = Measure(
                self.circuit,
                self.param_vals,
                [self.hamiltonian],
                [self.hamiltonian],
                num_shots=self.num_shots,
            )

            if should_print:
                print(
                    f"Optimize Parameters | Iteration: {iteration} | energy={measure.evs[self.hamiltonian]:.5f}, param_vals={self.param_vals}, grad={measure.grads[self.hamiltonian]}"
                )

            iteration += 1

            self.param_vals = self.optimizer.update(
                self.param_vals, measure.grads[self.hamiltonian]
            )

            if self.optimizer.is_converged(measure.grads[self.hamiltonian]):
                break

    def run(self: Self) -> None:
        self.circuit = self._make_ansatz()
        self.param_vals = (
            np.random.rand(len(self.circuit.parameters)).astype(np.float32) - 0.5
        )

        self.optimize_parameters()

        full_circuit = self.circuit.assign_parameters(
            {p: val for p, val in zip(self.circuit.parameters, self.param_vals)}
        )

        state_ev = exact_expectation_value(
            full_circuit, self.hamiltonian.operator_sparse
        )

        print(
            f"VQE[HF energy: {self.molecule.hf_energy},",
            f"Exact energy: {self.molecule.fci_energy},",
            f"Calculated energy: {state_ev},",
            f"Accuracy: {state_ev / self.molecule.fci_energy:.5%}]",
        )