from typing_extensions import Self, override
import numpy as np
import torch

from .optimizer import Optimizer
from ..utils.measure import Measure

class SGDOptimizer(Optimizer):
    """
    Performs SGD to optimize circuit parameters.
    """

    def __init__(self: Self, measure: Measure, step_size: float = 0.01) -> None:
        """
        step_size (float): learning rate for gradient descent updates.
        """
        super().__init__()
        self.measure = measure
        self.step_size = step_size
        self.values: list[float] = self.measure.param_values.copy().tolist()

    @override
    def update(self: Self) -> None:
        """
        Performs one step of gradient descent using gradient from measure class.
        """
        self.measure.param_values = np.array(self.values)

        self.measure.expectation_value = self.measure._calculate_expectation_value()
        self.measure.gradients = self.measure._calculate_gradients()

        grad = self.measure.gradients

        # new_params = old_params - step_size * grad
        updated_val = []
        for val, g in zip(self.values, grad):
            updated_val.append(val - self.step_size * g)

        self.values = updated_val