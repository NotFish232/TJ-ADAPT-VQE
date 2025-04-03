import numpy as np
from typing_extensions import Self, override

from ..utils.measure import Measure
from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Performs SGD to optimize circuit parameters.
    """

    def __init__(self: Self, step_size: float = 0.5, gradient_convergence_threshold: float = 0.01) -> None:
        """
        Args:
            step_size: float, the learning rate for gradient descent updates.
            gradient_convergence_threshold: float, the threshold that determines convergence
        """
        super().__init__(gradient_convergence_threshold=gradient_convergence_threshold)
        
        self.step_size = step_size
        
    @override
    def update(self: Self, param_vals: np.ndarray, measure: Measure) -> np.ndarray:
        """
        Performs one step of gradient descent using gradient from measure class.
        Uses standard gradient descent, traveling in the opposite direction by step_size
        """

        return param_vals - self.step_size * measure.gradients
