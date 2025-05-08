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


class Observable(ABC):
    """
    Base class for all observables
    """

    def __init__(self: Self, name: str, n_qubits: int) -> None:
        """
        Initializes the Observable

        Args:
            name: str, the name of the observable
            n_qubits: int, the number of qubits in the vector the observable is acting on
        """
        self.name = name
        self.n_qubits = n_qubits

        self.operator = self._make_operator()

    @abstractmethod
    def _make_operator(self: Self) -> LinearOp:
        """
        Generates the operator that is controlled by the observable
        Should be overriden in inherited classes
        """

        raise NotImplementedError()

    def __hash__(self: Self) -> int:
        return self.name.__hash__()

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Observable):
            return self.operator == other.operator

        raise NotImplementedError()

    def __str__(self: Self) -> str:
        return self.name

    def __repr__(self: Self) -> str:
        return self.__str__().__repr__()


class FermionObservable(Observable):
    """
    Creates an qiskit compatible observable from a fermion operator

    to define a new FermionObservable, inherit from this class and override
    the _make_fermion_operator method
    """

    def __init__(self: Self, name: str, n_qubits: int) -> None:
        """
        Args:
            name: str, the name of the observable
            n_qubits: int, the number of qubits in the vector the observable is acting on
        """

        super().__init__(name, n_qubits)

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
    def __init__(self: Self, sparse_pauli: SparsePauliOp, name: str):
        self.sparse_pauli = sparse_pauli

        super().__init__(name, sparse_pauli.num_qubits)

    @override
    def _make_operator(self: Self) -> LinearOp:
        return self.sparse_pauli


class NumberObservable(FermionObservable):
    """
    Observable for the Number Operator
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__("number_observable", n_qubits)

    @override
    def _make_fermion_operator(self: Self) -> FermionOperator:
        return sum(FermionOperator(f"{i}^ {i}") for i in range(self.n_qubits))  # type: ignore


class SpinZObservable(FermionObservable):
    """
    Observable for Spin Z
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__("spin_z_observable", n_qubits)

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
        super().__init__("spin_squared_observable", n_qubits)

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

        super().__init__("molecular_hamiltonian", self.molecule.n_qubits)

    @override
    def _make_fermion_operator(self: Self) -> InteractionOperator:
        return self.molecule.get_molecular_hamiltonian()
