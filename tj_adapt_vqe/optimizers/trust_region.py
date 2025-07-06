import numpy as np
from scipy.optimize import minimize
from typing_extensions import Callable, Self, override

from .optimizer import FunctionalOptimizer


class TrustRegionOptimizer(FunctionalOptimizer):
    """
    Inherits from `FunctionalOptimizer`. Trust region optimizer using scipy.
    """

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "trust_region_optimizer"

    @override
    def update(
        self: Self,
        param_vals: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        callback: Callable[[np.ndarray], None],
    ) -> None:
        """
        Optimizes the given function using scipy's trust region implementation.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            f (Callable[[np.ndarray], float]): A function f that maps from parameter values to function value.
            grad_f (Callable[[np.ndarray], np.ndarray]): A function grad_f that calculates the gradient of f at parameter values.
            callback (Callable[[np.ndarray], None]): A callback that takes the parameter values at each step.
        """

        minimize(f, param_vals, jac=grad_f, callback=callback, method="trust-constr")  # type: ignore
