import numpy as np
from scipy.optimize import minimize
from typing_extensions import Callable, Self, override

from .optimizer import FunctionalOptimizer


class Cobyla(FunctionalOptimizer):
    """
    Inherits from `FunctionalOptimizer`. Cobyla optimizer using scipy.
    """

    def __init__(
        self: Self,
    ) -> None:
        """
        Constructs an instance of Cobyla.

        Args:
            self (Self): A reference to the current class instance.
        """

        super().__init__("cobyla_optimizer")

    @override
    def update(
        self: Self,
        param_vals: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        callback: Callable[[np.ndarray], None],
    ) -> None:
        """
        Optimizes the given function using scipy's Cobyla implementation.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            f (Callable[[np.ndarray], float]): A function f that maps from parameter values to function value.
            grad_f (Callable[[np.ndarray], np.ndarray]): A function grad_f that calculates the gradient of f at parameter values.
            callback (Callable[[np.ndarray], None]): A callback that takes the parameter values at each step.
        """

        minimize(f, param_vals, callback=callback, method="COBYLA")  # type: ignore
