from itertools import combinations

from openfermion import MolecularData, jordan_wigner
from qiskit.circuit import Parameter, QuantumCircuit  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..ansatz.functional import (
    make_generalized_one_body_op,
    make_generalized_two_body_op,
)
from ..utils.conversions import openfermion_to_qiskit
from .pool import Pool


def make_parameterized_unitary_op(a: int, b: int, c: int, d: int) -> QuantumCircuit:
    """
    Creates a unitary operator that is parameterized by 3 operators and is acting on
    spatial orbitals p and q

    Args:
        p: int, first orbital to act on,
        q: int, second orbital to act on,
        layer: int, which layer unitary operator is on (only used for parameter naming)
    """

    # hard code the orbitals it maps to
    # orbitals will be mapped correctly when converting it to qiskit
    one_body_op = make_generalized_one_body_op(0, 1, 2, 3)
    two_body_op = make_generalized_two_body_op(0, 1, 2, 3)

    # apply the jordan wigner transformation and make operators strictly real
    one_body_op_jw = jordan_wigner(one_body_op)
    two_body_op_jw = jordan_wigner(two_body_op)

    # convert the jw representations to a qiskit compatible format (SparsePauliOp)
    one_body_op_qiskit = openfermion_to_qiskit(one_body_op_jw, 4)
    two_body_op_qiskit = openfermion_to_qiskit(two_body_op_jw, 4)

    # params = [Parameter(f"a{a}b{b}c{c}d{d}θ{i + 1}") for i in range(3)]
    params = [Parameter(f"[{a}, {b}]-[{c}, {d}]θ{i + 1}") for i in range(3)]

    qc = QuantumCircuit(4)

    # since qiskit PauliEvolutionGate adds the i to the exponentiation
    # similarly * -1 to counteract the PauliEvolutionGate
    # i * -i = 1
    gate_1 = PauliEvolutionGate(1j * one_body_op_qiskit, params[0])
    gate_2 = PauliEvolutionGate(1j * two_body_op_qiskit, params[1])
    gate_3 = PauliEvolutionGate(1j * one_body_op_qiskit, params[2])

    qc.append(gate_3, range(4))
    qc.append(gate_2, range(4))
    qc.append(gate_1, range(4))

    return qc


class UnrestrictedTUPSPool(Pool):
    """
    The Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    Considers each unitary_op as its own operator where the criteria is the sum of absolute values of gradients,
    Considers each combination of spatial orbitals rather that only adjacent ones
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        super().__init__(molecule)

        self.n_qubits = molecule.n_qubits
        self.n_spatials = molecule.n_qubits // 2

        self.operators, self.labels, self.orbitals = self.make_operators_and_labels()

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "unrestricted_tups_pool"

    def make_operators_and_labels(
        self: Self,
    ) -> tuple[list[list[LinearOp]], list[str], list[tuple[int, int, int, int]]]:
        operators = []
        labels = []
        orbitals = []

        p = [
            *combinations(
                range(
                    self.n_qubits - 1, self.n_qubits - self.molecule.n_electrons - 1, -1
                ),
                2,
            )
        ]  # HF
        for a, b in p:
            q = [
                *combinations(
                    range(self.molecule.n_qubits - self.molecule.n_electrons), 2
                )
            ]  # HF
            for c, d in q:
                one_body_op = make_generalized_one_body_op(a, b, c, d)
                two_body_op = make_generalized_two_body_op(a, b, c, d)

                one_body_op_qiskit = openfermion_to_qiskit(
                    jordan_wigner(one_body_op), self.n_qubits
                )
                two_body_op_qiskit = openfermion_to_qiskit(
                    jordan_wigner(two_body_op), self.n_qubits
                )

                operators.append(
                    [one_body_op_qiskit, two_body_op_qiskit, one_body_op_qiskit]
                )
                labels.append("U")
                orbitals.append((a, b, c, d))

        return operators, labels, orbitals

    @override
    def get_op(self: Self, idx: int) -> list[LinearOp]:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def get_exp_op(self: Self, idx: int) -> QuantumCircuit:
        a, b, c, d = self.orbitals[idx]
        u = make_parameterized_unitary_op(a, b, c, d)

        qc = QuantumCircuit(self.n_qubits)
        qc.append(u, [a, b, c, d])

        return qc

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
