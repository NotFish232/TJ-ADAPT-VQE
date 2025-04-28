from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Any, Self


class Optimizer(ABC):
    """
    Base Class that all other optimizers should inherit from
    """

    def __init__(
        self: Self, name: str, gradient_convergence_threshold: float = 0.025
    ) -> None:
        """
        Initializes the Optimizer

        Args:
            name: str, the name of the optimizer
            gradient_convergence_threshold: float, the threshold that qualifies for is_converged. Not used if Optimizer.is_converged is overrided
        """

        self.name = name
        self.gradient_convergence_threshold = gradient_convergence_threshold
    
    @abstractmethod
    def reset(self: Self) -> None:
        """
        Resets the optimizer state, for example needs to be compatible with a diff num of params
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self: Self, param_vals: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Performs a single step of optimization, returning the new param_vals,
        does NOT update either param_vals or measure in place.

        Args:
            param_vals: np.ndarray, a 1d numpy array with the current values of each param,
            gradient: np.ndarray, a numpy array which is the same dimension as param_vals and represents the gradient of each param_val
        """

        raise NotImplementedError()

    def is_converged(self: Self, gradient: np.ndarray) -> bool:
        """
        Returns whether or not the current optimizer is converged, the naive convergence criterion is whether the gradients all fall below some threshold
        """

        return bool(np.all(np.abs(gradient) < self.gradient_convergence_threshold))

    @abstractmethod
    def to_config(self: Self) -> dict[str, Any]:
        """
        Converts the Optimizer to a config that can be parsed back into an Optimizer
        """
        raise NotImplementedError()

    def __str__(self: Self) -> str:
        return self.name

    def __repr__(self: Self) -> str:
        return self.__str__().__repr__()
