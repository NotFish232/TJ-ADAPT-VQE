from itertools import combinations

from openfermion import FermionOperator, MolecularData, jordan_wigner
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Any, Self, override

from ..utils import openfermion_to_qiskit
from .pool import Pool


class FSDPool(Pool):
    """
    The fermionic single and double excitations pool.
    """

    def __init__(self: Self, molecule: MolecularData, n_excitations: int) -> None:
        super().__init__("fsd_pool", molecule)

        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.n_excitations = n_excitations

        self.operators, self.labels = self.make_operators_and_labels()

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        operators = []
        labels = []

        for n in range(1, self.n_excitations + 1):
            occupied = [*combinations(range(self.n_electrons), n)]
            virtual = [*combinations(range(self.n_electrons, self.n_qubits), n)]
            ops = [
                openfermion_to_qiskit(
                    jordan_wigner(
                        FermionOperator(
                            " ".join(f"{j}^" for j in v)
                            + " "
                            + " ".join(f"{j}" for j in o)
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
            "n_electrons": self.n_electrons,
            "n_excitations": self.n_excitations,
        }

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
