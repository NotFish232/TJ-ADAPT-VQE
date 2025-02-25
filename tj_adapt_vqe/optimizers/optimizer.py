from abc import ABC, abstractmethod
from typing_extensions import Self

from qiskit import QuantumCircuit  # type: ignore
from qiskit.circuit import Parameter  # type: ignore
from openfermion import MolecularData


class Optimizer(ABC):
    """
    Base Class that all other optimizers should inherit from
    Should probably take a reference to parameters to optimize and optimize them in place
    to prevent unncessary copying
    """

    def __init__(self: Self) -> None:
        """
        Initializes the VQE
        """

    def setup(
        self: Self,
        ansatz: QuantumCircuit,
        molecule: MolecularData,
        params: list[Parameter],
        initial_values: list[float],
        threshold: float = 10**-9,
    ) -> None:
        """
        ansatz: the ansatz circuit that the optimizer is optimizing
        params: the list of Qiskit Parameter objects in the circuit
        initial_values: an initial guess for what those parameters should be
        threshold: the amount that the energy must change less than to stop the optimization

        There needed to be a separate setup method from __init__ because we want to expose the details
        of the optimizer to the user (e.g. hyperparameters) when they create a VQE. For example:
            VQE(my_molecule, Optimizer(my_hyper_parameter))
        """
        self.ansatz = ansatz
        self.params = params
        self.values = initial_values
        self.threshold = threshold

    @abstractmethod
    def update(self: Self) -> None:
        """
        Performs one step of update
        Updates param values
        """

    def optimize(self: Self) -> tuple[float, list[float]]:
        """
        loop while not within threshold:
            update()
        return the final energy value, the list of parameters
        """

        return 0, []

    def assign_params(self: Self, qc: QuantumCircuit) -> QuantumCircuit:
        return qc.assign_parameters({p: v for p, v in zip(self.params, self.values)})
