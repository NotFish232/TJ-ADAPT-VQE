import numpy as np
from typing_extensions import Self, override

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(
        self: Self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
        super().__init__("Adam Optimizer")
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m: np.ndarray
        self.v: np.ndarray
        
        self.t = 0  # for bias correction

    @override
    def update(self: Self, param_vals: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Perform one update step using gradient
        Perform one update step using gradients from Measure (Adam optimizer).
        """

        if self.m is None:
            self.m = np.zeros_like(gradients.shape)
        if self.v is None:
            self.v = np.zeros_like(gradients.shape)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)


        updated_vals = param_vals - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_vals