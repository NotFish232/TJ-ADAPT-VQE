from typing_extensions import Self, override

from .pool import Pool


class QubitPool(Pool):
    """
    The qubit pool, which consists of the individual Pauli strings
    of the jordan wigner form of the operators in the GSD/QEB pools.
    """

    @override
    def make_operators(self: Self) -> None:
        pass
