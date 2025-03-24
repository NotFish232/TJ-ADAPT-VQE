from typing_extensions import Self, override
import numpy as np

from .optimizer import Optimizer
from ..utils.measure import Measure

class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(
        self: Self,
        measure: Measure,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
        super().__init__()
        self.measure = measure
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.values: list[float] = self.measure.param_values.copy().tolist()

        self.m = np.zeros_like(self.measure.param_values)
        self.v = np.zeros_like(self.measure.param_values)
        self.t = 0  # for bias correction

    @override
    def update(self: Self) -> None:
        """
        Perform one update step using gradient from Measure.
        """
        # update param_values in Measure
        self.measure.param_values = np.array(self.values)

        self.measure.expectation_value = self.measure._calculate_expectation_value()
        self.measure.gradients = self.measure._calculate_gradients()

        grad = self.measure.gradients

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        updated_val = []
        for val, m_h, v_h in zip(self.values, m_hat, v_hat):
            updated = val - self.learning_rate * m_h / (np.sqrt(v_h) + self.epsilon)
            updated_val.append(updated)

        self.values = updated_val
