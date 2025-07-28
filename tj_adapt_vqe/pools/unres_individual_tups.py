from itertools import permutations

from openfermion import jordan_wigner
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..ansatz.functional import (
    make_generalized_one_body_op,
    make_generalized_two_body_op,
)
from ..utils.conversions import openfermion_to_qiskit
from ..utils.molecules import Molecule
from .pool import Pool


class UnresIndividualTUPSPool(Pool):
    """
    The Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    Considers each unitary_op as its own operator where the criteria is the sum of absolute values of gradients,
    Considers each combination of spatial orbitals rather that only adjacent ones
    """

    def __init__(self: Self, molecule: Molecule) -> None:
        super().__init__(molecule)

        self.n_qubits = molecule.data.n_qubits

        self.operators, self.labels = self.make_operators_and_labels()

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "unrestricted_individual_tups_pool"

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        operators = []
        labels = []

        one_body_orbitals = [*permutations(range(self.n_qubits), 4)]
        two_body_orbitals = [*permutations(range(self.n_qubits), 4)]

        i = 0
        while i < len(one_body_orbitals):
            a, b, c, d = one_body_orbitals[i]

            # can swap (a, c) <=> (b, d)
            # can also swap (a, b) <=> (c, d)
            for p, q, r, s in [(b, a, d, c), (c, d, a, b), (d, c, b, a)]:
                one_body_orbitals.remove((p, q, r, s))

            i += 1

        i = 0
        while i < len(two_body_orbitals):
            a, b, c, d = two_body_orbitals[i]

            # can swap (a) <=> (b) and (c) <=> (d)
            # can also swap (a, b) <=> (c, d)
            for p, q, r, s in [
                (b, a, c, d),
                (a, b, d, c),
                (b, a, d, c),
                (c, d, a, b),
                (d, c, a, b),
                (c, d, b, a),
                (d, c, b, a),
            ]:
                two_body_orbitals.remove((p, q, r, s))

            i += 1

        one_bodies = [
            make_generalized_one_body_op(a, b, c, d) for a, b, c, d in one_body_orbitals
        ]
        one_labels = [f"κ(1)[{a},{b},{c},{d}]" for a, b, c, d in one_body_orbitals]
        two_bodies = [
            make_generalized_two_body_op(a, b, c, d) for a, b, c, d in two_body_orbitals
        ]
        two_labels = [f"κ(2)[{a},{b},{c},{d}]" for a, b, c, d in two_body_orbitals]

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
