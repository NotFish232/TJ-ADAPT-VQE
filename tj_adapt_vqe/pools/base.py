from abc import ABC, abstractmethod
from typing_extensions import Self

from openfermion import MolecularData

class Pool(ABC):
    """
    Base class for all pools
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        """
        Takes a molecule and defines an operator pool for it
        Constructor should (probably) precompute possible operators
        """
        self.molecule = molecule
        self.make_operators()

    @abstractmethod
    def make_operators(self: Self) -> None:
        pass

    """
    Some combination of getters / get imp op / get exps etc 
    """
