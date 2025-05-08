# TODO: FIXME, weird compatibility issues with jax and multithreading
import warnings

import numpy as np
import optax  # type: ignore
from typing_extensions import Any, Self, override

from .optimizer import GradientOptimizer

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=r".*os\.fork\(\) was called.*"
)


class LBFGS(GradientOptimizer):
    """
    Inherits from `GradientOptimizer`. Quasi-Newton BFGS optimizer using jax.
    """

    def __init__(
        self: Self,
        lr: float = 0.1,
        memory_size: int = 10,
        grad_conv_threshold: float = 0.01,
    ) -> None:
        """
        Constructs an instance of LBFGS.

        Args:
            self (Self): A reference to the current class instance.
            lr (float, optional): The learning rate for updates. Defaults to 0.1.
            memory_size (int, optional): The memory size for LBFGS. Defaults to 10.
            grad_conv_threshold (float, optional): The gradient threshold for convergence. Defaults to 0.01.
        """

        super().__init__("lbfgs_optimizer", grad_conv_threshold)

        self.lr = lr
        self.memory_size = memory_size

        self.reset()

    @override
    def reset(self: Self) -> None:
        """
        Resets the state by intializing a new lbfgs instance.

        Args:
            self (Self): A reference to the current class instance.
        """

        self.lbfgs = optax.scale_by_lbfgs(self.memory_size)
        self.state: optax.OptState = None

    @override
    def update(self: Self, param_vals: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Performs a single update step of LBFGS returning the new parameter values.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            gradients (np.ndarray): The gradient with respect to each parameter values.

        Returns:
            np.ndarray: The new parameter values.
        """

        if self.state is None:
            self.state = self.lbfgs.init(param_vals)

        updates, self.state = self.lbfgs.update(gradients, self.state, param_vals)

        return param_vals - self.lr * np.array(updates)

    @override
    def to_config(self: Self) -> dict[str, Any]:
        """
        Conveerts the optimizer state into a configuration.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            dict[str, Any]: The configuration of the optimizer.
        """
        
        base_config = super().to_config()

        return base_config | {
            "lr": self.lr,
            "memory_size": self.memory_size,
        }
