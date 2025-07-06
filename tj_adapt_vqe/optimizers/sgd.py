import numpy as np
from typing_extensions import Self, override

from .optimizer import GradientOptimizer


class SGDOptimizer(GradientOptimizer):
    """
    Inherits from `GradientOptimizer`. An implementation of Stochastic Gradient Descent.
    """

    def __init__(
        self: Self, lr: float = 0.01, grad_conv_threshold: float = 0.01
    ) -> None:
        """
        Constructs an instance of SGD.

        Args:
            self (Self): A reference to the current class instance.
            lr (float, optional): The learning rate for updates. Defaults to 0.01.
            grad_conv_threshold (float, optional): The gradient threshold for convergence. Defaults to 0.01.
        """

        self.lr = lr
        self.grad_conv_threshold = grad_conv_threshold

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "sgd_optimizer"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["lr", "grad_conv_threshold"]

    @override
    def update(self: Self, param_vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Performs one step of update to param_vals. Simple update by moving in the opposite direction
        of the gradient with a step size of the learning rate.

        Args:
            self (Self): A rerference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            grad (np.ndarray): The gradients of each parameter value.

        Returns:
            np.ndarray: The new parameter values.
        """

        return param_vals - self.lr * grad

    @override
    def is_converged(self: Self, grad: np.ndarray) -> bool:
        """
        Simple implementation of is_converged with gradient.
        """

        return np.max(np.abs(grad)) < self.grad_conv_threshold
