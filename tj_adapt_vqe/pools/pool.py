from abc import ABC, abstractmethod

from openfermion import MolecularData
from qiskit import QuantumCircuit  # type: ignore
from typing_extensions import Self


class Pool(ABC):
    """
    Base class for all pools
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        """
        Takes a molecule and defines an operator pool for it
        Constructor should (probably) precompute possible operators
        """
        self.molecule = molecule
        self.make_operators()

    @abstractmethod
    def make_operators(self: Self) -> None:
        """
        The method that generates the pool operators for the molecule
        """

    """
    Some combination of getters / get imp op / get exps etc 
    """

    @abstractmethod
    def get_circuit_operator(self: Self, i: int) -> QuantumCircuit:
        """
        Returns the circuit representation of an operator at index i in the operators array attribute
        Pools may store an internal representation different from a QuantumCircuit, therefore needing conversion
        """
