from openfermion import MolecularData, jordan_wigner
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..ansatz.functional import (
    make_one_body_op,
    make_parameterized_unitary_op,
    make_two_body_op,
)
from ..utils.conversions import openfermion_to_qiskit  # type: ignore
from .pool import Pool


class MultiTUPSPool(Pool):
    """
    The Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    Considers each unitary_op as its own operator where the criteria is the sum of absolute values of gradients,
    Considers each combination of spatial orbitals rather that only adjacent ones
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        super().__init__(molecule)

        self.n_qubits = molecule.n_qubits
        self.n_spatials = molecule.n_qubits // 2

        self.operators, self.labels, self.spatial_orbitals = (
            self.make_operators_and_labels()
        )

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "multi_tups_pool"

    def make_operators_and_labels(
        self: Self,
    ) -> tuple[list[list[LinearOp]], list[str], list[tuple[int, int]]]:
        operators = []
        labels = []
        spatial_orbitals = []

        for p_1 in range(self.n_spatials):
            for p_2 in range(p_1 + 1, self.n_spatials):
                one_body_op = make_one_body_op(p_1, p_2)
                two_body_op = make_two_body_op(p_1, p_2)

                one_body_op_qiskit = openfermion_to_qiskit(
                    jordan_wigner(one_body_op), self.n_qubits
                )
                two_body_op_qiskit = openfermion_to_qiskit(
                    jordan_wigner(two_body_op), self.n_qubits
                )

                operators.append(
                    [one_body_op_qiskit, two_body_op_qiskit, one_body_op_qiskit]
                )
                labels.append(f"U_{p_1 + 1}_{p_2 + 1}")
                spatial_orbitals.append((p_1, p_2))

        return operators, labels, spatial_orbitals

    @override
    def get_op(self: Self, idx: int) -> list[LinearOp]:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def get_exp_op(self: Self, idx: int) -> QuantumCircuit:
        p_1, p_2 = self.spatial_orbitals[idx]
        u = make_parameterized_unitary_op(p_1 + 1, p_2 + 1)

        qc = QuantumCircuit(self.n_qubits)
        qc.append(u, [2 * p_1, 2 * p_1 + 1, 2 * p_2, 2 * p_2 + 1])

        return qc

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
