from openfermion import MolecularData
from typing_extensions import Self, override

from tj_adapt_vqe.pools import Pool
from tj_adapt_vqe.vqe import VQE


class ADAPTVQE(VQE):
    """
    Class implementing the ADAPT-VQE algorithm
    """

    def __init__(self: Self, molecule: MolecularData, pool: Pool) -> None:
        """
        Initializes the ADAPTVQE object
        Arguments:
            molecule (MolecularData): the molecular data that is used for the adapt vqe
            pool (Pool): the pool that the ADAPTVQE uses to form the Ansatz
        """
        self.pool = pool


  