from itertools import combinations

from openfermion import FermionOperator, MolecularData, jordan_wigner
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..utils import openfermion_to_qiskit
from .pool import Pool


class FSD(Pool):
    """
    The fermionic single and double excitations pool.
    """

    def __init__(self: Self, molecule: MolecularData, n_excitations: int) -> None:
        self.n_excitations = n_excitations
        super().__init__(molecule)

    @override
    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        operators = []
        labels = []

        for n in range(1, self.n_excitations + 1):
            occupied = [*combinations(range(0, self.n_electrons), n)]
            virtual = [*combinations(range(self.n_electrons, self.n_qubits), n)]
            ops = [
                openfermion_to_qiskit(
                    jordan_wigner(
                        FermionOperator(
                            " ".join(f"{j}^" for j in v)
                            + " "
                            + " ".join(str(j) for j in o)
                        )
                    ),
                    self.n_qubits,
                )
                for o in occupied
                for v in virtual
            ]
            ops = [1j * (op - op.conjugate().transpose()).simplify() for op in ops]
            operators += ops
            labels += [f"o{o}v{v}" for v in virtual for o in occupied]

        return operators, labels
