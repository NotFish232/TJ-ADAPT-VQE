from abc import ABC, abstractmethod

from openfermion import MolecularData
from qiskit.circuit import Gate, Parameter, QuantumCircuit  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..utils.serializable import Serializable


class Pool(Serializable, ABC):
    """
    Inherits from `abc.ABC`. Base class for all other pools.
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        """
        Constructs an instance of a Pool.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule associated with the pool.
        """

        self.molecule = molecule

    @staticmethod
    @override
    def _type() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "pool"

    @abstractmethod
    def get_op(self: Self, idx: int) -> LinearOp | list[LinearOp]:
        """
        Gets the operator assocaited with the index from the pool.
        Method can either return a LinearOp or a list[LinearOp], if
        there are multiple matrices associated with a single operator.

        Args:
            self (Self): A reference to the current class instance.
            idx (int): The idx of the operator to be returned.

        Returns:
            LinearOp|list[LinearOp]: The associated operator or list of operators.
        """

        pass

    @abstractmethod
    def get_label(self: Self, idx: int) -> str:
        """
        Gets the label associated with the operator at the idx from the pool.

        Args:
            idx: int, the idx of the operator in the pool

        Returns:
            str: The label associated with the operator from the pool.
        """

        pass

    def get_exp_op(self: Self, idx: int) -> Gate | QuantumCircuit:
        """
        Gets the evolved operator which is the parameterized gate / quantum circuit
        that is added to the qiskit circuit. Default implementation is simply evolving
        a `LinearOp` with the `PauliEvolutionGate`.

        Args:
            self (Self): A reference to the current class instance.
            idx (int): The idx of the associated operator in the pool.

        Raises:
            NotImplementedError: If `get_op(...)` returned a list of operators.

        Returns:
            Gate | QuantumCircuit: The gate to add to the qiskit circuit.
        """

        op = self.get_op(idx)

        if isinstance(op, LinearOp):
            return PauliEvolutionGate(1j * op, Parameter("ϴ"))

        raise NotImplementedError()

    @abstractmethod
    def __len__(self: Self) -> int:
        """
        Abstract implementation of the length dunder that subclasses must override.
        Should be the number of operators in the pool so `get_op(i)` for i < len
        is always valid.

        Args:
            self (Self): A reference to the current instance.

        Returns:
            int: The number of operators within the pool.
        """

        pass
