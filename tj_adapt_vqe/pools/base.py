from typing_extensions import Self
from openfermion import MolecularData


class Pool:
    """
    Base class for all pools
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        """
        Takes a molecule and defines an operator pool for it
        Constructor should (probably) precompute possible operators
        """
        pass

    """
    Some combination of getters / get imp op / get exps etc 
    """
