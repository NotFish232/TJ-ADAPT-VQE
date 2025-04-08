from abc import ABC, abstractmethod

from openfermion import MolecularData
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Any, Self


class Pool(ABC):
    """
    Base class for all pools
    """

    def __init__(self: Self, name: str, molecule: MolecularData) -> None:
        """
        Takes a molecule and defines an operator pool for it
        Constructor should (probably) precompute possible operators
        """

        self.name = name
        self.molecule = molecule

        self.n_electrons = molecule.n_electrons
        self.n_qubits = molecule.n_qubits

        self.operators, self.labels = self.make_operators_and_labels()
    
    @abstractmethod
    def to_config(self: Self) -> dict[str, Any]:
        """
        Converts the pool to a config that can be used to recreate the pool from the dictionary of arguments
        """

        raise NotImplementedError()

    @abstractmethod
    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        """
        The method that generates the pool operators for the molecule as well as a label for each operator
        Should return a tuple of two equal length lists, where each element in the first list
        is the pool operator and each element in the second list is the label for that operator
        """

        raise NotImplementedError()
    
    def __str__(self: Self) -> str:
        return self.name
    
    def __repr__(self: Self) -> str:
        return self.__str__().__repr__()
