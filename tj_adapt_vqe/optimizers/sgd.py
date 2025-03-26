import numpy as np
from typing_extensions import Self, override

from ..utils.measure import Measure
from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Performs SGD to optimize circuit parameters.
    """

    def __init__(self: Self, step_size: float = 0.01) -> None:
        """
        measure (Measure): an instance of the Measure class that will compute gradients.
        step_size (float): learning rate for gradient descent updates.
        """
        super().__init__()
        
        self.step_size = step_size
        
    @override
    def update(self: Self, param_vals: np.ndarray, measure: Measure) -> np.ndarray:
        """
        Performs one step of gradient descent using gradient from measure class.
        Returns the updated parameter values as a new NumPy array.
        """

        gradients = measure.gradients

        updated_vals = param_vals - self.step_size * np.array(gradients)

        return np.array(updated_vals)
