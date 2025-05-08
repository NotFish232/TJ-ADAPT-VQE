from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from typing_extensions import Any, Callable, Self


class OptimizerType(Enum):
    Gradient = 0
    NonGradient = 1
    Hybrid = 2


class Optimizer(ABC):
    """
    Base Class that all other optimizers should inherit from
    """

    def __init__(self: Self, name: str, type: OptimizerType) -> None:
        """
        Initializes the Optimizer

        Args:
            name: str, the name of the optimizer
            type: OptimizerType, the type of the optimizer
        """

        self.name = name
        self.type = type


    def reset(self: Self) -> None:
        """
        Resets the optimizer state, for example needs to be compatible with a diff num of params
        """
        pass


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


class GradientOptimizer(Optimizer):
    """
    Base class for gradient optimizers
    """

    def __init__(
        self: Self, name: str, gradient_convergence_threshold: float = 0.01
    ) -> None:
        """
        Initializes the Optimizer

        Args:
            name: str, the name of the optimizer
            gradient_convergence_threshold: float, the threshold that qualifies for is_converged. Not used if Optimizer.is_converged is overrided
        """

        super().__init__(name, OptimizerType.Gradient)

        self.gradient_convergence_threshold = gradient_convergence_threshold


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


class NonGradientOptimizer(Optimizer):
    """
    Base class for non gradient optimizers
    """

    def __init__(
        self: Self, name: str
    ) -> None:
        """
        Initializes the Optimizer

        Args:
            name: str, the name of the optimizer
        """

        super().__init__(name, OptimizerType.NonGradient)



    @abstractmethod
    def update(self: Self, param_vals: np.ndarray, f: Callable[[np.ndarray], float]) -> np.ndarray:
        """
        Performs a single step of optimization, returning the new param_vals,
        does NOT update either param_vals or measure in place.

        Args:
            param_vals: np.ndarray, a 1d numpy array with the current values of each param,
            gradient: np.ndarray, a numpy array which is the same dimension as param_vals and represents the gradient of each param_val
        """

        raise NotImplementedError()

    
    @abstractmethod
    def is_converged(self: Self) -> bool:
        """
        Returns whether the optimzier is converged
        """


class HybridOptimizer(Optimizer):
    """
    Base class for optimizers that require both gradients and evaluation function
    """

    def __init__(
        self: Self, name: str
    ) -> None:
        """
        Initializes the Optimizer

        Args:
            name: str, the name of the optimizer
        """

        super().__init__(name, OptimizerType.Hybrid)



    @abstractmethod
    def update(self: Self, param_vals: np.ndarray, gradient: np.ndarray, f: Callable[[np.ndarray], float]) -> np.ndarray:
        """
        Performs a single step of optimization, returning the new param_vals,
        does NOT update either param_vals or measure in place.

        Args:
            param_vals: np.ndarray, a 1d numpy array with the current values of each param,
            gradient: np.ndarray, a numpy array which is the same dimension as param_vals and represents the gradient of each param_val
        """

        raise NotImplementedError()

    
    @abstractmethod
    def is_converged(self: Self, gradient: np.ndarray) -> bool:
        raise NotImplementedError()