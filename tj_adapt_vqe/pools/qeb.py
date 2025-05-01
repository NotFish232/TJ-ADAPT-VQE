from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from .pool import Pool


class QEB(Pool):
    """
    Qubit excitations pool. Equivalent to the generalized excitations pools,
    but without the antisymmetry Z strings in the jordan wigner representation.
    """

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        return [], []
