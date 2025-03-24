from typing_extensions import Self, override
import numpy as np
import torch
from scipy.optimize import minimize

from .optimizer import Optimizer, QuantumCircuit
from ..utils.measure import Measure

class BFGS(Optimizer):
    """
    Quasi-Newton BFGS optimizer using scipy.
    """

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def update(self: Self) -> None:
        self.bfgs()

    def bfgs(self: Self) -> None:
        ''''
        runs BFGS from the current parameter values to convergence.
        '''

        # MIGHT HAVE TO CHANGE LATER
        def cost_fn(param_values: np.ndarray) -> float:
            qc = torch.tensor(param_values, dtype=torch.float32)
            return self.measure.get_energy(qc).item()

        init_guess = np.array(self.values, dtype=float)

        output = minimize(cost_fn, init_guess, method="BFGS")

        self.values = output.x.tolist()
