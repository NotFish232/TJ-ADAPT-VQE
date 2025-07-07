from itertools import combinations

from openfermion import jordan_wigner
from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..ansatz.functional import (
    make_generalized_one_body_op,
    make_generalized_two_body_op,
    make_parameterized_unitary_op,
)
from ..utils.conversions import openfermion_to_qiskit, prepend_params
from ..utils.molecules import Molecule
from .pool import Pool


class UnrestrictedTUPSPool(Pool):
    """
    The Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    Considers each unitary_op as its own operator where the criteria is the sum of absolute values of gradients,
    Considers each combination of spatial orbitals rather that only adjacent ones
    """

    def __init__(self: Self, molecule: Molecule) -> None:
        super().__init__(molecule)

        self.n_qubits = molecule.data.n_qubits
        self.n_electrons = molecule.data.n_electrons

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
                range(self.n_qubits - 1, self.n_qubits - self.n_electrons - 1, -1),
                2,
            )
        ]  # HF
        for a, b in p:
            q = [*combinations(range(self.n_qubits - self.n_electrons), 2)]  # HF
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
        u = make_parameterized_unitary_op()
        u = prepend_params(u, f"[{a}, {b}]-[{c}, {d}]")

        qc = QuantumCircuit(self.n_qubits)
        qc.append(u, [a, b, c, d])

        return qc

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
