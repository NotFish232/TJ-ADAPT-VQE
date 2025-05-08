import numpy as np
from typing_extensions import Any, Callable, Self, override

from .optimizer import HybridOptimizer


class LevenbergMarquardt(HybridOptimizer):
    """
    Levenberg-Marquardt-style optimizer for scalar-valued objective functions.
    """

    def __init__(
        self: Self,
        damping: float = 1e-2,
        max_iter: int = 100,
        tol: float = 1e-6,
        target: float = 0.0,
    ) -> None:
        """
        Args:
            damping: float, initial damping factor (λ),
            max_iter: int, maximum number of updates per call,
            tol: float, convergence tolerance on parameter updates,
            target: float, target expectation value to reach (residual = value - target)
        """
        super().__init__("levenberg_marquardt_optimizer")

        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.target = target
    
    @override
    def reset(self: Self) -> None:
        pass

    @override
    def update(self: Self, param_vals: np.ndarray, gradients: np.ndarray, f: Callable[[np.ndarray], float]) -> np.ndarray:
        """
        Perform a few steps of damped Gauss-Newton-style updates using gradient info.
        """
        x = param_vals

        # TODO: FIXME (are you sure you need f_x and target?)
        # f_x = # measure.expectation_value
        # grad = gradients

        # residual = f_x - self.target
        # grad = np.asarray(grad, dtype=np.float64).flatten()
        # H_approx = np.outer(grad, grad)

        # # Levenberg-Marquardt damping
        # A = H_approx + self.damping * np.eye(len(x))
        # g = grad * residual

        # dx = np.linalg.solve(A, -g)

        # x += dx

        return x

    @override
    def to_config(self: Self) -> dict[str, Any]:
        return {}
