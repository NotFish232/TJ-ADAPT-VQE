from abc import ABC, abstractmethod

from typing_extensions import Self


class Observable(ABC):
    def __init__(self: Self):
        pass

    @abstractmethod
    def evaluate(self: Self, bitstring: str) -> int:
        """
        Evaluates the operator for an (eigen)state represented by a bitstring
        Returns the eigenvalue for that bitstring
        """

    def expectation_value(self: Self, counts: dict[str, int]) -> float:
        """
        Returns the expectation value for a results dict from Qiskit
        """
        return -1

    def uncertainty(self: Self, counts: dict[str, int]) -> float:
        """
        Returns the uncertainty for a results dict from Qiskit
        """
        return -1