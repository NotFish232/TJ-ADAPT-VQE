from typing import Any

import numpy as np
from typing_extensions import Self, override

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(
        self: Self,
        lr: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        gradient_convergence_threshold: float = 0.01,
    ) -> None:
        """
        Args:
            lr: float, the learning rate for gradient descent updates,
            beta_1: float, beta 1 for the Adam algorithm,
            beta_2: float, beta 2 for hte Adam algorithm,
            gradient_convergence_threshold: float, the threshold that determines convergence
        """
        super().__init__("Adam Optimizer", gradient_convergence_threshold)
        
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        

        self.reset()
   
  
    
    @override
    def reset(self: Self) -> None:
        self.m: np.ndarray = None # type: ignore
        self.v: np.ndarray = None # type: ignore
        self.t = 0

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
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradients
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradients ** 2)

        m_cor = self.m / (1 - self.beta_1 ** self.t)
        v_cor = self.v / (1 - self.beta_2 ** self.t)


        new_vals = param_vals - self.lr * m_cor / (np.sqrt(v_cor) + 1e-8)

        return new_vals

    @override
    def to_config(self: Self) -> dict[str, Any]:
        """
        Defines the config for a Adam optimizer
        """
        return {
            "name": self.name,
            "lr": self.lr,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "gradient_convergence_threshold": self.gradient_convergence_threshold
        }