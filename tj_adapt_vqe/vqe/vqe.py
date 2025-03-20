from openfermion import MolecularData
from qiskit import QuantumRegister  # type: ignore
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit.circuit.library import EfficientSU2  # type: ignore
from typing_extensions import Self

from tj_adapt_vqe.optimizers.optimizer import Optimizer  # type: ignore


class VQE:
    """
    Class implementing the variational quantum eigensolver (VQE) algorithm
    """

    def __init__(
        self: Self,
        molecule: MolecularData,
        optimizer: Optimizer,
        num_shots: int = 1000
    ) -> None:
        """
        Initializes starting Ansatz (num qubits are calculated for MolecularData object)
        Initializes optimizer with the starting conditions including parameters and intiial state
        num_shots is num_shots to use for gradient calculations in the Measure instance
        """
        self.molecule = molecule
        self.optimizer = optimizer
        self.num_shots = num_shots

        self.ansatz = self.make_initial_ansatz()


    def make_initial_ansatz(self: Self) -> QuantumCircuit:
        """
        Constructs the parameterized Ansatz circuit to be optimized using the Ansatz circuit created by initialize_state
        """
        # TODO FIXME: currently just using a premade ansatz as the starting state
        # update with a better educates guess likely using HF approximation
        quantum_circuit = EfficientSU2(self.molecule.n_qubits)

        return quantum_circuit

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
