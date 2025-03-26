from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self

from tj_adapt_vqe.utils import Measure


class Optimizer(ABC):
    """
    Base Class that all other optimizers should inherit from
    Should probably take a reference to parameters to optimize and optimize them in place
    to prevent unncessary copying
    """

    def __init__(self: Self) -> None:
        """
        Initializes the Optimizer
        """

    @abstractmethod
    def update(self: Self, param_vals: np.ndarray, measure: Measure) -> np.ndarray:
        """
        Performs one step of update
        Updates param values
        """

