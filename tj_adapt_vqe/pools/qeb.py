from typing_extensions import Self, override

from tj_adapt_vqe.pools.pool import Pool


class QEB(Pool):
    """
    Qubit excitations pool. Equivalent to the generalized excitations pools,
    but without the antisymmetry Z strings in the jordan wigner representation.
    """

    @override
    def make_operators(self: Self) -> None:
        pass
