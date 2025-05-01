from qiskit.quantum_info.operators import LinearOp  # type: ignore
from typing_extensions import Self, override

from .pool import Pool


class QubitPool(Pool):
    """
    The qubit pool, which consists of the individual Pauli strings
    of the jordan wigner form of the operators in the GSD/QEB pools.
    """

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        return [], []
