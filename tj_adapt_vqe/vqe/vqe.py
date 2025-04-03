import numpy as np
from openfermion import MolecularData, jordan_wigner, get_sparse_operator
from qiskit.circuit import Parameter, QuantumCircuit  # type: ignore
from typing_extensions import Self

from tj_adapt_vqe.optimizers.optimizer import Optimizer
from tj_adapt_vqe.utils import openfermion_to_qiskit
from tj_adapt_vqe.utils.measure import Measure, exact_expectation_value


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

        self.param_vals = np.array([0.0])

    def _make_initial_circuit(self: Self) -> QuantumCircuit:
        """
        Constructs the parameterized Ansatz circuit to be optimized using the Ansatz circuit created by initialize_state
        """
        # TODO FIXME: currently just using a premade ansatz as the starting state
        # update with a better educates guess likely using HF approximation

        theta = Parameter("θ")

        qc = QuantumCircuit(4)

        qc.x(2)
        qc.x(3)

        qc.barrier()

        qc.cx(1, 0)
        qc.cx(3, 2)
        qc.x(0)
        qc.x(2)
        qc.cx(1, 3)
        qc.ry(theta / 4, 1)
        qc.h(0)
        qc.cx(1, 0)
        qc.h(2)
        qc.ry(-theta / 4, 1)
        qc.cx(1, 2)
        qc.ry(theta / 4, 1)
        qc.cx(1, 0)
        qc.h(3)
        qc.ry(-theta / 4, 1)
        qc.cx(1, 3)
        qc.ry(theta / 4, 1)
        qc.cx(1, 0)
        qc.ry(-theta / 4, 1)
        qc.cx(1, 2)
        qc.ry(theta / 4, 1)
        qc.cx(1, 0)
        qc.ry(-theta / 4, 1)
        qc.h(2)
        qc.h(0)
        qc.rz(np.pi / 2, 3)
        qc.cx(1, 3)
        qc.rz(-np.pi / 2, 1)
        qc.rz(np.pi / 2, 3)
        qc.ry(np.pi / 2, 3)
        qc.x(0)
        qc.x(2)
        qc.cx(1, 0)
        qc.cx(3, 2)

        # Apply a barrier
        qc.barrier()

        return qc

    def optimize_parameters(self: Self) -> None:
        """
        Performs a single iteration step of the vqe, stopping when the provided Optimizer's stopping condition has been reached
        """

        # TODO: add logging

        iteration = 1

        while True:
            measure = Measure(
                self.circuit, self.param_vals, self.molecular_hamiltonian_qiskit
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

        state_ev = exact_expectation_value(full_circuit, self.molecular_hamiltonian_sparse)

        print(
            f"HF energy: {self.molecule.hf_energy}, Exact energy: {self.molecule.fci_energy}, Calculated energy: {state_ev}"
        )
