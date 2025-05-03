from abc import ABC, abstractmethod

from openfermion import MolecularData
from qiskit.circuit import Gate, Parameter  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore
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

    @abstractmethod
    def get_op(self: Self, idx: int) -> LinearOp:
        """
        Gets the operator at the idx from the pool

        Args:
            idx: int, the idx of the operator in the pool
        """
        raise NotImplementedError()

    @abstractmethod
    def get_label(self: Self, idx: int) -> str:
        """
        Gets the label assocaited with the opeartor at the idx from the pool

        Args:
            idx: int, the idx of the operator in the pool
        """
        raise NotImplementedError()

    def get_exp_op(self: Self, idx: int) -> Gate:
        """
        Gets the exponentiated operator assocaited with that idx in the pool
        This has a generic implementation of exp(A * theta), but can be overriden for
        pools that require a different exponentiation strategy.

        Args
            idx: int, the idx of the operator in the pool
        """
        return PauliEvolutionGate(1j * self.get_op(idx), Parameter("ϴ"))

    @abstractmethod
    def to_config(self: Self) -> dict[str, Any]:
        """
        Converts the pool to a config that can be used to recreate the pool from the dictionary of arguments
        """

        raise NotImplementedError()

    @abstractmethod
    def __len__(self: Self) -> int:
        """
        the length attribute should be overrided for each subclass
        """
        raise NotImplementedError()

    def __str__(self: Self) -> str:
        return self.name

    def __repr__(self: Self) -> str:
        return self.__str__().__repr__()
