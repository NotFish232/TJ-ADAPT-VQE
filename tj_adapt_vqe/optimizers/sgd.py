import numpy as np
from typing_extensions import Any, Self, override

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Performs SGD to optimize circuit parameters.
    """

    def __init__(
        self: Self, lr: float = 0.1, gradient_convergence_threshold: float = 0.01
    ) -> None:
        """
        Args:
            lr: float, the learning rate for gradient descent updates.
            gradient_convergence_threshold: float, the threshold that determines convergence
        """
        super().__init__("SGD Optimizer", gradient_convergence_threshold)

        self.lr = lr

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

        return param_vals - self.lr * gradients

    @override
    def to_config(self: Self) -> dict[str, Any]:
        """
        Defines the config for a SGD optimizer which is simply just the learning rate
        """
        return {
            "name": self.name,
            "lr": self.lr,
            "gradient_convergence_threshold": self.gradient_convergence_threshold,
        }
