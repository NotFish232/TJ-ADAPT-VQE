from abc import ABC, abstractmethod

from openfermion import (
    FermionOperator,
    InteractionOperator,
    MolecularData,
    get_sparse_operator,
    jordan_wigner,
)
from typing_extensions import Self, override
from qiskit.quantum_info.operators import SparsePauliOp

from ..utils.molecules import openfermion_to_qiskit


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

        self.operator = self._create_operator()
        self.operator_sparse = get_sparse_operator(self.operator)
        self.operator_qiskit = openfermion_to_qiskit(
            jordan_wigner(self.operator), self.n_qubits
        )

    @abstractmethod
    def _create_operator(self: Self) -> FermionOperator | InteractionOperator:
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
        return self.name.__repr__()


class SparsePauliObservable(Observable):
    def __init__(self: Self, operator: SparsePauliOp, name: str, n_qubits: int):
        self.operator_qiskit = operator
        self.name = name
        self.n_qubits = n_qubits

    @override
    def _create_operator(self: Self) -> None:
        pass


class NumberObservable(Observable):
    """
    Observable for the Number Operator
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__("Number Observable", n_qubits)

    @override
    def _create_operator(self: Self) -> FermionOperator:
        return sum(FermionOperator(f"{i}^ {i}") for i in range(self.n_qubits))  # type: ignore


class SpinZObservable(Observable):
    """
    Observable for Spin Z
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__("Spin Z Observable", n_qubits)

    @override
    def _create_operator(self: Self) -> FermionOperator:
        # TODO: FIXME this is flipped but it gives the right answer, why?
        return (1 / 2) * sum(
            FermionOperator(f"{i}^ {i}", -1 if i % 2 == 0 else 1)
            for i in range(self.n_qubits)
        )  # type: ignore


class SpinSquaredObservable(Observable):
    """
    Observable for Spin Squared
    """

    def __init__(self: Self, n_qubits: int) -> None:
        super().__init__("Spin Squared Observable", n_qubits)

    @override
    def _create_operator(self: Self) -> FermionOperator:
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


class HamiltonianObservable(Observable):
    """
    Observable for the Hamiltonian
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        self.molecule = molecule

        super().__init__("Molecular Hamiltonian", self.molecule.n_qubits)

    @override
    def _create_operator(self: Self) -> InteractionOperator:
        return self.molecule.get_molecular_hamiltonian()
