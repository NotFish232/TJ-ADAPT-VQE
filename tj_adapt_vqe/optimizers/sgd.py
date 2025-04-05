import numpy as np
from typing_extensions import Self, override

from ..utils import Measure
from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Performs SGD to optimize circuit parameters.
    """

    def __init__(self: Self, learning_rate: float = 0.5, gradient_convergence_threshold: float = 0.01) -> None:
        """
        Args:
            learning_rate: float, the learning rate for gradient descent updates.
            gradient_convergence_threshold: float, the threshold that determines convergence
        """
        super().__init__(gradient_convergence_threshold=gradient_convergence_threshold)
        
        self.learning_rate = learning_rate
        
    @override
    def update(self: Self, param_vals: np.ndarray, measure: Measure) -> np.ndarray:
        """
        Performs one step of gradient descent using gradient from measure class.
        Uses standard gradient descent, traveling in the opposite direction by step_size
        """

        return param_vals - self.learning_rate * measure.gradients
