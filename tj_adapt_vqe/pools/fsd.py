from typing_extensions import Self, override

from tj_adapt_vqe.pools.pool import Pool


class FSD(Pool):
    """
    The fermionic single and double excitations pool.
    """

    @override
    def make_operators(self: Self) -> None:
        pass
