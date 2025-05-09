# TODO: FIXME, weird compatibility issues with jax and multithreading
import warnings

import numpy as np
import optax  # type: ignore
from typing_extensions import Any, Self, override, Callable
from .optimizer import FunctionalOptimizer
from scipy.optimize import minimize


class LBFGS(FunctionalOptimizer):
    """
    Inherits from `FunctionalOptimizer`. Quasi-Newton BFGS optimizer using scipy.
    """

    def __init__(
        self: Self,
    ) -> None:
        """
        Constructs an instance of LBFGS.

        Args:
            self (Self): A reference to the current class instance.
        """

        super().__init__("lbfgs_optimizer")

    
    def update(
        self: Self,
        param_vals: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        callback: Callable[[np.ndarray], None],
    ) -> None:
        """
        Optimizes the given function using scipy's LBFGS implementation.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            f (Callable[[np.ndarray], float]): A function f that maps from parameter values to function value.
            grad_f (Callable[[np.ndarray], np.ndarray]): A function grad_f that calculates the gradient of f at parameter values.
            callback (Callable[[np.ndarray], None]): A callback that takes the parameter values at each step.
        """

        minimize(f, param_vals, jac=grad_f, callback=callback, method="L-BFGS-B") # type: ignore
