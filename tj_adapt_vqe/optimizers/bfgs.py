from typing_extensions import Self, override
import numpy as np
from scipy.optimize import minimize

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
    def update(self: Self) -> None:
        self.bfgs()

    def bfgs(self: Self) -> None:
        """
        Runs BFGS from the current parameter values to convergence.
        """

        def cost_fn(param_values: np.ndarray) -> float:
            # Update the measure with new params
            self.measure.param_values = param_values
            return self.measure._calculate_expectation_value()

        init_guess = np.array(self.values, dtype=float)

        output = minimize(cost_fn, init_guess, method="BFGS")

        self.values = output.x.tolist()
