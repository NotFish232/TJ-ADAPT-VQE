from itertools import combinations

from openfermion import MolecularData, jordan_wigner, FermionOperator, normal_ordered
from qiskit.circuit import Parameter, QuantumCircuit  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Any, Self, override

from ..utils.conversions import openfermion_to_qiskit  # type: ignore
from .pool import Pool


def normalize_op(operator: FermionOperator) -> FermionOperator:
    """
    Normalizes a symbolic operator by making the magnitudes of the coefficients sum to 0
    """

    return operator / sum(abs(c) for c in operator.terms.values())

def make_one_body_op(a: int, b: int, c: int, d: int) -> FermionOperator:
    """
    Returns a generalized one body fermionic operator acting on spin orbitals a & b, and c & d
    """
    e_pq = FermionOperator(f"{a}^ {c}") + FermionOperator(f"{b}^ {d}")
    e_qp = FermionOperator(f"{c}^ {a}") + FermionOperator(f"{d}^ {b}")

    op = e_pq - e_qp

    return normalize_op(normal_ordered(op))

def make_two_body_op(a: int, b: int, c: int, d: int) -> FermionOperator:
    """
    Returns a generalized two body fermionic operator acting on spin orbitals a & b, and c & d
    """
    # e_pq = FermionOperator(f"{2 * p}^ {2 * q}") + FermionOperator(
    #     f"{2 * p + 1}^ {2 * q + 1}"
    # )
    # e_qp = FermionOperator(f"{2 * q}^ {2 * p}") + FermionOperator(
    #     f"{2 * q + 1}^ {2 * p + 1}"
    # )
    e_pq = FermionOperator(f"{a}^ {c}") + FermionOperator(f"{b}^ {d}")
    e_qp = FermionOperator(f"{c}^ {a}") + FermionOperator(f"{d}^ {b}")

    op = e_pq**2 - e_qp**2

    return normalize_op(normal_ordered(op))


class UnresIndividualTUPSPool(Pool):
    """
    The Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    Considers each unitary_op as its own operator where the criteria is the sum of absolute values of gradients,
    Considers each combination of spatial orbitals rather that only adjacent ones
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        super().__init__("unres_individual_tups_pool", molecule)

        self.n_qubits = molecule.n_qubits
        self.n_spatials = molecule.n_qubits // 2

        self.operators, self.labels = self.make_operators_and_labels()

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        operators = []
        labels = []

        choose_four = [*combinations(range(self.n_qubits), 4)]
        one_bodies = [
            make_one_body_op(a, b, c, d)
            for a, b, c, d in choose_four
        ]
        one_labels = [
            f"κ(1)[{a},{b},{c},{d}]"
            for a, b, c, d in choose_four
        ]
        two_bodies = [
            make_two_body_op(a, b, c, d)
            for a, b, c, d in choose_four
        ]
        two_labels = [
            f"κ(2)[{a},{b},{c},{d}]"
            for a, b, c, d in choose_four
        ]

        operators = one_bodies + two_bodies
        operators = [
            openfermion_to_qiskit(jordan_wigner(o), self.molecule.n_qubits)
            for o in operators
        ]
        labels = one_labels + two_labels

        print(labels)
        print(len(labels))
        return operators, labels

    @override
    def get_op(self: Self, idx: int) -> LinearOp:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def to_config(self: Self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "n_spatials": self.n_spatials,
        }

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
