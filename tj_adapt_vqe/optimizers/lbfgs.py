# TODO: FIXME, weird compatibility issues with jax and multithreading
import warnings

import numpy as np
import optax  # type: ignore
from typing_extensions import Any, Self, override

from .optimizer import Optimizer

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=r".*os\.fork\(\) was called.*"
)


class LBFGS(Optimizer):
    """
    Quasi-Newton BFGS optimizer using jax.
    """

    def __init__(
        self: Self,
        lr: float = 0.5,
        memory_size: int = 20,
        gradient_convergence_threshold: float = 0.01,
    ) -> None:
        """
        Args:
            lr: float, learning_rate for updates,
            memory_size: int, memory_size for LBFGS, defaults to 20,
            gradient_convergence_threshold: float, the threshold that determines convergence
        """
        super().__init__("BFGS Optimizer", gradient_convergence_threshold)

        self.lr = lr
        self.memory_size = memory_size

        self.reset()

    @override
    def reset(self: Self) -> None:
        """
        Resets internal state
        """
        self.lbfgs = optax.scale_by_lbfgs(self.memory_size)
        self.state: optax.OptState = None

    @override
    def update(self: Self, param_vals: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Run BFGS optimization starting from param_vals and return optimized parameters.
        """

        if self.state is None:
            self.state = self.lbfgs.init(param_vals)

        updates, self.state = self.lbfgs.update(gradients, self.state, param_vals)

        return param_vals - self.lr * np.array(updates)

    @override
    def to_config(self: Self) -> dict[str, Any]:
        """
        Defines the config for a BFGS optimizer
        """
        return {
            "name": self.name,
            "lr": self.lr,
            "memory_size": self.memory_size,
            "gradient_convergence_threshold": self.gradient_convergence_threshold,
        }
