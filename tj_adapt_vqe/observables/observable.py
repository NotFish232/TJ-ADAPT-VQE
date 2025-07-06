from abc import ABC, abstractmethod

from openfermion import (
    FermionOperator,
    InteractionOperator,
    MolecularData,
    get_sparse_operator,
    jordan_wigner,
)
from qiskit.quantum_info.operators import SparsePauliOp  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..utils.conversions import openfermion_to_qiskit
from ..utils.serializable import Serializable


class Observable(Serializable, ABC):
    """
    Base class for all observables
    """

    def __init__(self: Self, n_qubits: int) -> None:
        """
        Initializes the Observable

        Args:
            n_qubits: int, the number of qubits in the vector the observable is acting on
        """
        self.n_qubits = n_qubits

        self.operator = self._make_operator()

    @staticmethod
    @override
    def _type() -> str:
        """
        Returns the type of this class. Used in `Serializable`.
        """

        return "observable"

    @abstractmethod
    def _make_operator(self: Self) -> LinearOp:
        """
        Generates the operator that is controlled by the observable
        Should be overriden in inherited classes
        """

        raise NotImplementedError()

    def __hash__(self: Self) -> int:
        return str(self).__hash__()

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Observable):
            return self.operator == other.operator

        raise NotImplementedError()


class FermionObservable(Observable):
    """
    Creates an qiskit compatible observable from a fermion operator

    to define a new FermionObservable, inherit from this class and override
    the _make_fermion_operator method
    """

    def __init__(self: Self, n_qubits: int) -> None:
        """
        Args:
            n_qubits: int, the number of qubits in the vector the observable is acting on
        """

        super().__init__(n_qubits)

    @abstractmethod
    def _make_fermion_operator(self: Self) -> FermionOperator | InteractionOperator:
        """
        New method that should be overriden for fermion operators
        """

        raise NotImplementedError()

    @override
    def _make_operator(self: Self) -> LinearOp:
        self.fermion_operator = self._make_fermion_operator()
        self.operator_sparse = get_sparse_operator(self.fermion_operator)

        return openfermion_to_qiskit(
            jordan_wigner(self.fermion_operator), self.n_qubits
        )


class SparsePauliObservable(Observable):
    """
    Creates an Observable that wraps a qiskit `SparsePauliOp`. Used in pools
    """

    def __init__(self: Self, sparse_pauli: SparsePauliOp) -> None:
        self.sparse_pauli = sparse_pauli

        super().__init__(sparse_pauli.num_qubits)

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of the class. Although SparsePauliObservable is never serialized.
        """

        return "sparse_pauli_observable"

    @override
    def _make_operator(self: Self) -> LinearOp:
        return self.sparse_pauli


class NumberObservable(FermionObservable):
    """
    Observable for the Number Operator
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__(n_qubits)

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "number_observable"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["n_qubits"]

    @override
    def _make_fermion_operator(self: Self) -> FermionOperator:
        return sum(FermionOperator(f"{i}^ {i}") for i in range(self.n_qubits))  # type: ignore


class SpinZObservable(FermionObservable):
    """
    Observable for Spin Z
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__(n_qubits)

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "spin_z_observable"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["n_qubits"]

    @override
    def _make_fermion_operator(self: Self) -> FermionOperator:
        return (1 / 2) * sum(
            FermionOperator(f"{i}^ {i}", 1 if i % 2 == 0 else -1)
            for i in range(self.n_qubits)
        )  # type: ignore


class SpinSquaredObservable(FermionObservable):
    """
    Observable for Spin Squared
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__(n_qubits)

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "spin_squared_observable"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["n_qubits"]

    @override
    def _make_fermion_operator(self: Self) -> FermionOperator:
        spin_z = (1 / 2) * sum(
            FermionOperator(f"{i}^ {i}", 1 if i % 2 == 0 else -1)
            for i in range(self.n_qubits)
        )
        spin_plus = sum(
            FermionOperator(f"{i}^ {i + 1}") for i in range(0, self.n_qubits, 2)
        )
        spin_minus = sum(
            FermionOperator(f"{i + 1}^ {i}") for i in range(0, self.n_qubits, 2)
        )

        return spin_minus * spin_plus + spin_z * (spin_z + 1)  # type: ignore


class HamiltonianObservable(FermionObservable):
    """
    Observable for the Hamiltonian
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        self.molecule = molecule

        super().__init__(self.molecule.n_qubits)

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "hamiltonian_observable"

    @override
    def _make_fermion_operator(self: Self) -> InteractionOperator:
        return self.molecule.get_molecular_hamiltonian()
