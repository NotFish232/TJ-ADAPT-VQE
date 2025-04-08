from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from .pool import Pool


class GSD(Pool):
    """
    The generalized singles and doubles pool. Differs from fermionic SD,
    which only includes excitations from occupied to virtual orbitals,
    by including excitations from virtual to virtual, occupied to occupied, and virtual to occupied.
    """

    @override
    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        return [], []
