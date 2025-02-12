from abc import ABC, abstractmethod
from qiskit import QuantumCircuit  # type: ignore
from typing_extensions import Self
from qiskit.circuit import Parameter  # type: ignore


class Optimizer(ABC):
    """
    Base Class that all other optimizers should inherit from
    Should probably take a reference to parameters to optimize and optimize them in place
    to prevent unncessary copying
    """

    def __init__(
        self: Self, params: list[Parameter], initial_values: list[float]
    ) -> None:
        self.params = params
        self.values = initial_values

    @abstractmethod
    def update(self: Self) -> None:
        """
        Performs one step of update
        Updates param values
        """

    def assign_params(self: Self, qc: QuantumCircuit) -> QuantumCircuit:
        return qc.assign_parameters({p: v for p, v in zip(self.params, self.values)})
