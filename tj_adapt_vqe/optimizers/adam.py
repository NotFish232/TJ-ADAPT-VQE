# tj_adapt_vqe/optimizers/adam.py

from typing_extensions import Self, override
import numpy as np
import torch

from .optimizer import Optimizer, QuantumCircuit
from ..utils.measure import Measure

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
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0  # for bias correction

    @override
    def update(self: Self) -> None:
        """
        Perform one update step using gradient from Measure.get_grad.
        """
        qc = self.assign_params(self.ansatz_circuit)

        # CHANGE HERE
        grad = self.measure.get_grad(torch.tensor(self.values, dtype=torch.float32))
        grad = np.array(grad, dtype=float)

        # moment vectors
        if self.m is None:
            self.m = np.zeros_like(grad)
        if self.v is None:
            self.v = np.zeros_like(grad)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # bias-corrected estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # update rule
        updated_val = []
        for val, m_h, v_h in zip(self.values, m_hat, v_hat):
            updated = val - self.learning_rate * m_h / (np.sqrt(v_h) + self.epsilon)
            updated_val.append(updated)

        self.values = updated_val
        

