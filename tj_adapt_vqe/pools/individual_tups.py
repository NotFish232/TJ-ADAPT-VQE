from openfermion import jordan_wigner
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..ansatz.functional import make_one_body_op, make_two_body_op
from ..utils.conversions import openfermion_to_qiskit
from ..utils.molecules import Molecule
from .pool import Pool


class IndividualTUPSPool(Pool):
    """
    The Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    only considers Individual one body and two body operators with a single exponentiation and param
    """

    def __init__(self: Self, molecule: Molecule) -> None:
        super().__init__(molecule)

        self.n_qubits = molecule.data.n_qubits
        self.n_spatials = self.n_qubits // 2

        self.operators, self.labels = self.make_operators_and_labels()

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "individual_tups_pool"

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        operators = []
        labels = []

        one_bodies = [
            make_one_body_op(i, j)
            for i in range(self.n_spatials)
            for j in range(i + 1, self.n_spatials)
        ]
        one_labels = [
            f"κ(1)[{i},{j}]"
            for i in range(self.n_spatials)
            for j in range(i + 1, self.n_spatials)
        ]
        two_bodies = [
            make_two_body_op(i, j)
            for i in range(self.n_spatials)
            for j in range(i + 1, self.n_spatials)
        ]
        two_labels = [
            f"κ(2)[{i},{j}]"
            for i in range(self.n_spatials)
            for j in range(i + 1, self.n_spatials)
        ]

        operators = one_bodies + two_bodies
        operators = [
            openfermion_to_qiskit(jordan_wigner(o), self.n_qubits) for o in operators
        ]
        labels = one_labels + two_labels

        return operators, labels

    @override
    def get_op(self: Self, idx: int) -> LinearOp:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
