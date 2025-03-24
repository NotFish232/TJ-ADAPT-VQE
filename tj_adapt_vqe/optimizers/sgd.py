from typing_extensions import Self, override
import numpy as np
import torch

from .optimizer import Optimizer
from ..utils.measure import Measure

class SGDOptimizer(Optimizer):
    """
    Performs SGD to optimize circuit parameters.
    """

    def __init__(self: Self, step_size: float = 0.01) -> None:
        """
        step_size (float): learning rate for gradient descent updates.
        """
        super().__init__()
        self.step_size = step_size

    @override
    def update(self: Self) -> None:
        """
        Performs one step of gradient descent using gradient from measure class.
        """
        # assign current parameter vals to circuit
        #qc = self.assign_params(self.ansatz_circuit)

        # assume measure.get_grad returns a torch tensor
        grad = self.measure.get_grad(torch.tensor(self.values, dtype=torch.float32))

        # new_params = old_params - step_size * grad
        updated_val = []
        for val, g in zip(self.values, grad):
            updated_val.append(val - self.step_size * g.item())

        self.values = updated_val

