import numpy as np
from typing_extensions import Self, override

from .optimizer import GradientOptimizer


class AdamOptimizer(GradientOptimizer):
    """
    Inherits from `GradientOptimizer`. An implementation of the Adam Optimizer.
    """

    def __init__(
        self: Self,
        lr: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        grad_conv_threshold: float = 0.01,
    ) -> None:
        """
        Constructs an instance of Adam.

        Args:
            self (Self): A reference to the current class instance.
            lr (float, optional): The learning rate for updates. Defaults to 0.01.
            beta_1 (float, optional): The beta 1 hyperparameter for Adam. Defaults to 0.9.
            beta_2 (float, optional): The beta 2 hyperparameter for Adam. Defaults to 0.999.
            grad_conv_threshold (float, optional): The gradient threshold for convergence. Defaults to 0.01.
        """

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.grad_conv_threshold = grad_conv_threshold

        self.reset()

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "adam_optimizer"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["lr", "beta_1", "beta_2", "grad_conv_threshold"]

    @override
    def reset(self: Self) -> None:
        """
        Resets the optimizer state by zeroring out momentum, variance, and timestamp.

        Args:
            self (Self): A reference to the current class instance.
        """

        self.m: np.ndarray = None  # type: ignore
        self.v: np.ndarray = None  # type: ignore
        self.t = 0

    @override
    def update(self: Self, param_vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Performs a single update step of adam. Calculates new values of momentum and
        variance using the beta values. Corrects momentum and variance using the time values.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            grad (np.ndarray): The gradient with respect to each parameter value.

        Returns:
            np.ndarray: The new parameter values.
        """

        if self.m is None:
            self.m = np.zeros_like(grad.shape)
        if self.v is None:
            self.v = np.zeros_like(grad.shape)

        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad**2)

        m_cor = self.m / (1 - self.beta_1**self.t)
        v_cor = self.v / (1 - self.beta_2**self.t)

        new_vals = param_vals - self.lr * m_cor / (np.sqrt(v_cor) + 1e-8)

        return new_vals

    @override
    def is_converged(self: Self, grad: np.ndarray) -> bool:
        """
        Simple implementation of is_converged with gradient.
        """

        return np.max(np.abs(grad)) < self.grad_conv_threshold
