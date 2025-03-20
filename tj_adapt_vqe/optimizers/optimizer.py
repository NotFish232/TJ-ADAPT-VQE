from abc import ABC, abstractmethod

from openfermion import MolecularData
from qiskit import QuantumCircuit  # type: ignore
from qiskit.circuit import Parameter  # type: ignore
from typing_extensions import Self

from tj_adapt_vqe.utils import Measure


class Optimizer(ABC):
    """
    Base Class that all other optimizers should inherit from
    Should probably take a reference to parameters to optimize and optimize them in place
    to prevent unncessary copying
    """

    def __init__(self: Self) -> None:
        """
        Initializes the Optimizer
        """

    def setup(
        self: Self,
        measure: Measure,
        molecule: MolecularData,
        params: list[Parameter],
        initial_values: list[float],
        threshold: float = 10**-9,
    ) -> None:
        """
        measure: measures the ansatz circuit for the Optimizer
        params: the list of Qiskit Parameter objects in the circuit
        initial_values: an initial guess for what those parameters should be
        threshold: the amount that the energy must change less than to stop the optimization

        There needed to be a separate setup method from __init__ because we want to expose the details
        of the optimizer to the user (e.g. hyperparameters) when they create a VQE. For example:
            VQE(my_molecule, Optimizer(my_hyper_parameter))
        """
        self.measure = measure
        self.params = params
        self.values = initial_values
        self.threshold = threshold

    @abstractmethod
    def update(self: Self) -> None:
        """
        Performs one step of update
        Updates param values
        """

    def assign_params(self: Self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        A helper to assign the parameters to the circuit
        qc: the quantum circuit to assign self.values for self.params to
        Returns the quantum circuit with the set parameter values
        """
        return qc.assign_parameters({p: v for p, v in zip(self.params, self.values)})
