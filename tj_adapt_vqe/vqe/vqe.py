import numpy as np
from openfermion import MolecularData, jordan_wigner
from qiskit.circuit import Parameter, QuantumCircuit  # type: ignore
from typing_extensions import Self

from tj_adapt_vqe.optimizers.optimizer import Optimizer
from tj_adapt_vqe.utils import openfermion_to_qiskit


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

        self.molecular_hamiltonian = molecule.get_molecular_hamiltonian()
        self.molecular_hamiltonian_jw = jordan_wigner(self.molecular_hamiltonian)
        self.molecular_hamiltonian_qiskit = openfermion_to_qiskit(
            self.molecular_hamiltonian_jw, molecule.n_qubits
        )

        self.optimizer = optimizer

        self.num_shots = num_shots

        self.circuit = self._make_initial_circuit()
        self.param_values = np.zeros(
            len(self.circuit.parameters)
        )  # np.random.rand(len(self.circuit.parameters)) - 0.5

    def _make_initial_circuit(self: Self) -> QuantumCircuit:
        """
        Constructs the parameterized Ansatz circuit to be optimized using the Ansatz circuit created by initialize_state
        """
        # TODO FIXME: currently just using a premade ansatz as the starting state
        # update with a better educates guess likely using HF approximation

        theta = Parameter("θ")

        qc = QuantumCircuit(4)

        qc.x(0)
        qc.x(1)

        qc.h(0)
        qc.h(1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.cx(2, 3)

        qc.ry(theta, 0)

        return qc

    def optimize(self: Self) -> None:
        """
        optimizes the Ansatz parameters. Needs to also be able to use a custom optimzer from the optimizers module
        while not converged:
            # probably make a method for gradient calculation
            self.optimizer.update()
            if self._update_ansatz is not None:
                self._update_ansatz()
        Returns some metrics of training, i.e. how the parameters are updating, what the current ground energy is etc
        """

    def run(self: Self) -> float:
        """
        Runs the VQE. Executes (a) initialize_state (b) make_ansatz, then (c) optimize
        returns the final energy value. perhaps should also return the ansatz circuit with or w/o the parameters
        """

        return 0
