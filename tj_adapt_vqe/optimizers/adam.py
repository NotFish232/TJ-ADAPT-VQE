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
    def update(self: Self, param_vals: np.ndarray, measure: Measure) -> np.ndarray:
        """
        Perform one update step using gradient
        Perform one update step using gradients from Measure (Adam optimizer).
        """
        gradients = measure._calculate_gradients()
        grad = np.array(gradients)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        updated_vals = param_vals - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_vals
