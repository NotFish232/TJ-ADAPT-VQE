import numpy as np
from typing_extensions import Any, Self, override

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Performs SGD to optimize circuit parameters.
    """

    def __init__(self: Self, learning_rate: float = 0.5, gradient_convergence_threshold: float = 0.025) -> None:
        """
        Args:
            learning_rate: float, the learning rate for gradient descent updates.
            gradient_convergence_threshold: float, the threshold that determines convergence
        """
        super().__init__("SGD Optimizer", gradient_convergence_threshold)
        
        self.learning_rate = learning_rate
    
    @override
    def reset(self: Self) -> None:
        """
        SGD does not need to do anything since no state
        """
        pass
        
    @override
    def update(self: Self, param_vals: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Performs one step of gradient descent using gradient from measure class.
        Uses standard gradient descent, traveling in the opposite direction by step_size
        """

        return param_vals - self.learning_rate * gradients
    
    @override
    def to_config(self: Self) -> dict[str, Any]:
        """
        Defines the config for a SGD optimizer which is simply just the learning rate
        """
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "gradient_convergence_threshold": self.gradient_convergence_threshold
        }
