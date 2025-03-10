from abc import ABC, abstractmethod
from typing_extensions import Self

from openfermion import MolecularData
from qiskit.circuit import QuantumCircuit  # type: ignore

from tj_adapt_vqe.optimizers.optimizer import Optimizer  # type: ignore


class VQE(ABC):
    """
    Class implementing the variational quantum eigensolver (VQE) algorithm
    """

    def __init__(
        self: Self,
        molecule: MolecularData,
        optimizer: Optimizer,
        threshold: float = 10**-9,
    ) -> None:
        """
        Initializes starting Ansatz (constructor probably needs to take num qubits?)
        Maybe take a callback or something if we want a better starting point
        Also need either molecule / moleculer hamiltonian to actually calculate
        the expected value on our Ansatze (william says molecule is nice please supply that)
        """
        self.molecule = molecule
        self.optimizer = optimizer

        self.ansatz = self.initialize_state()
        self.ansatz = self.make_ansatz()

    def initialize_state(self: Self) -> QuantumCircuit:
        """
        Creates the ansatz attribute of the class, and adds some gates to initialize the Hartree Fock state from the molecule attribute
        """

    @abstractmethod
    def make_ansatz(self: Self) -> QuantumCircuit:
        """
        Makes the parameterized Ansatz circuit to be optimized using the Ansatz circuit created by initialize_state
        """

    def optimize(self: Self) -> None:
        """
        optimizes the Ansatz parameters. Needs to also be able to use a custom optimzer from the optimizers module
        while not converged:
            # probably make a method for gradient calculation
            self.optimizer.update()
            if self._update_ansatz is not None:
                self._update_ansatz()
        Returns some metrics of trainiRleng, i.e. how the parameters are updating, what the current ground energy is etc
        Returns some metrics of training, i.e. how the parameters are updating, what the current ground energy is etc
        """

    def run(self: Self) -> float:
        """
        Runs the VQE. Executes (a) initialize_state (b) make_ansatz, then (c) optimize
        returns the final energy value. perhaps should also return the ansatz circuit with or w/o the parameters
        """

        return 0
