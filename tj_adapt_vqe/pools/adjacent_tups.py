from openfermion import MolecularData, jordan_wigner
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Any, Self, override

from ..utils.ansatz import (
    make_one_body_op,
    make_parameterized_unitary_op,
    make_two_body_op,
)
from ..utils.conversions import openfermion_to_qiskit  # type: ignore
from .pool import Pool


class AdjacentTUPSPool(Pool):
    """
    The Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    Considers each unitary_op as its own operator where the criteria is the sum of absolute values of gradients,
    only considers adjacent spatial orbitals
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        super().__init__("Adjacent TUPS Pool", molecule)

        self.n_qubits = molecule.n_qubits
        self.n_spatials = molecule.n_qubits // 2

        self.operators, self.labels = self.make_operators_and_labels()

    def make_operators_and_labels(
        self: Self,
    ) -> tuple[list[list[LinearOp]], list[str]]:
        operators = []
        labels = []

        for p in range(self.n_spatials - 1):
            one_body_op = make_one_body_op(p, p + 1)
            two_body_op = make_two_body_op(p, p + 1)

            one_body_op_qiskit = openfermion_to_qiskit(jordan_wigner(one_body_op), self.n_qubits)
            two_body_op_qiskit = openfermion_to_qiskit(jordan_wigner(two_body_op), self.n_qubits)

            operators.append([one_body_op_qiskit, two_body_op_qiskit, one_body_op_qiskit])
            labels.append(f"U_{p + 1}_{p + 2}")

        return operators, labels

    @override
    def get_op(self: Self, idx: int) -> list[LinearOp]:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def get_exp_op(self: Self, idx: int) -> QuantumCircuit:
        p = idx
        u = make_parameterized_unitary_op(p + 1, p + 2)

        qc = QuantumCircuit(self.n_qubits)
        qc.append(u, range(2 * p, 2 * p + 4))

        return qc

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
