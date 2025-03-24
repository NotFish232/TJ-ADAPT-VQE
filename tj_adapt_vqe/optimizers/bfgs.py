import numpy as np

from scipy.optimize import minimize  # type: ignore
from typing_extensions import Self, override

from ..utils.measure import Measure
from .optimizer import Optimizer


class BFGS(Optimizer):
    """
    Quasi-Newton BFGS optimizer using scipy.
    """

    def __init__(self: Self, measure: Measure) -> None:
        super().__init__()
        
    @override
    def update(self: Self, param_vals: np.ndarray, measure: Measure) -> np.ndarray:
        """
        Run BFGS optimization starting from param_vals and return optimized parameters.
        """
        def cost_fn(param_values: np.ndarray, measure: Measure) -> float:
            measure.param_values = param_values
            return measure._calculate_expectation_value()

        init_val = np.array(param_vals, dtype=float)
        output = minimize(cost_fn, init_val, method="BFGS")

        return output.x
