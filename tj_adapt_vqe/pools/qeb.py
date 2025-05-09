from itertools import combinations

from openfermion import MolecularData, QubitOperator
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Any, Self, override

from ..utils import openfermion_to_qiskit
from .pool import Pool


class QEBPool(Pool):
    """
    Qubit excitations pool. Equivalent to the generalized excitations pools,
    but without the antisymmetry Z strings in the jordan wigner representation.
    """

    def __init__(self: Self, molecule: MolecularData, n_excitations: int) -> None:
        super().__init__("qeb_pool", molecule)

        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.n_excitations = n_excitations

        self.operators, self.labels = self.make_operators_and_labels()

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        """
        The method that generates the pool operators for the molecule as well as a label for each operator
        Should return a tuple of two equal length lists, where each element in the first list
        is the pool operator and each element in the second list is the label for that operator
        """
        operators = []
        labels = []
        for n in range(1, self.n_excitations + 1):
            occupied = [*combinations(range(self.n_qubits), n)]
            virtual = [*combinations(range(self.n_qubits), n)]
            ops = [
                openfermion_to_qiskit(
                    2 ** (-2 * n)
                    * (
                        self._concat_ops(
                            [
                                QubitOperator(f"X{j}") + 1j * QubitOperator(f"Y{j}")
                                for j in v
                            ]
                        )
                        * self._concat_ops(
                            [
                                QubitOperator(f"X{j}") - 1j * QubitOperator(f"Y{j}")
                                for j in o
                            ]
                        )
                    ),
                    self.n_qubits,
                )
                for o in occupied
                for v in virtual
            ]
            ops = [(op - op.conjugate().transpose()).simplify() for op in ops]
            operators += ops
            labels += [f"o{o}v{v}" for v in virtual for o in occupied]

        rem = set()
        for j, op in enumerate(operators):
            if all(j == 0 for j in op.coeffs):
                rem.add(j)
        operators = [k for j, k in enumerate(operators) if j not in rem]
        labels = [k for j, k in enumerate(labels) if j not in rem]

        return operators, labels

    def _concat_ops(self: Self, ops: list[QubitOperator]) -> QubitOperator:
        k = ops[0]
        for i in range(1, len(ops)):
            k = k * ops[i]
        return k

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
            "n_electrons": self.n_electrons,
            "n_excitations": self.n_excitations,
        }

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
