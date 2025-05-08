import numpy as np
from typing_extensions import Any, Self, override

from .optimizer import GradientOptimizer


class SGD(GradientOptimizer):
    """
    Inherits from `GradientOptimizer`. An implementation of Stochastic Gradient Descent.
    """

    def __init__(
        self: Self, lr: float = 0.1, grad_conv_threshold: float = 0.01
    ) -> None:
        """
        Constructs an instance of SGD.

        Args:
            self (Self): A reference to the current class instance.
            lr (float, optional): The learning rate for updates. Defaults to 0.1.
            grad_conv_threshold (float, optional): The gradient threshold for convergence. Defaults to 0.01.
        """

        super().__init__("sgd_optimizer", grad_conv_threshold)

        self.lr = lr

    @override
    def update(self: Self, param_vals: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Performs one step of update to param_vals. Simple update by moving in the opposite direction
        of the gradient with a step size of the learning rate.

        Args:
            self (Self): A rerference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            gradients (np.ndarray): The gradients of each parameter value.

        Returns:
            np.ndarray: The new parameter values.
        """

        return param_vals - self.lr * gradients

    @override
    def to_config(self: Self) -> dict[str, Any]:
        """
        Converts the Optimizer to a configuration.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            dict[str, Any]: The configuration asscoaited with the current optimizer.
        """

        base_config = super().to_config()

        return base_config | {
            "lr": self.lr,
        }
