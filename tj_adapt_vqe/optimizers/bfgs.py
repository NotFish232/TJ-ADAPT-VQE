from typing_extensions import Self, override
import numpy as np
from scipy.optimize import minimize # type: ignore

from .optimizer import Optimizer
from ..utils.measure import Measure

class BFGS(Optimizer):
    """
    Quasi-Newton BFGS optimizer using scipy.
    """

    def __init__(self: Self, measure: Measure) -> None:
        super().__init__()
        self.measure = measure
        self.values: list[float] = self.measure.param_values.tolist()

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
