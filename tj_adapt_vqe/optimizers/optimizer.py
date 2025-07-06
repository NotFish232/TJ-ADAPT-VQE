from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Callable, Self, override

from ..utils.serializable import Serializable


class Optimizer(Serializable, ABC):
    """
    Inherits from `Serializable` and `abc.ABC`. A base class that each other optimizer should inherit from.
    Subclasses should define the methods `update(...)` and `is_converged(...)` with different arguments.
    """

    @staticmethod
    @override
    def _type() -> str:
        """
        Returns the type of this class. Used in `Serializable`.
        """

        return "optimizer"

    def reset(self: Self) -> None:
        """
        Resets the state of the optimizer. For example, optimizers may be reused with different
        numbers of parameters between calls to `update(...)`. All mutable state used between
        calls to `update(...)` should bd reset here. Subclasses should override this method if
        they require any mutable state.

        Args:
            self (Self): A reference to the current class instance.
        """

        pass


class GradientOptimizer(Optimizer):
    """
    Inherits from `Optimizer`. Base class for all optimizers requiring gradients for calls to `update()`
    and `is_converged()`.
    """

    @abstractmethod
    def update(self: Self, param_vals: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Performs a single step of optimization, returning the new param_vals,
        does not update param_vals in place. An abstract method in which subclasses
        should override with their implementation of a single update step.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): A 1d array with the current parameter values, the same shape as gradient.
            grad (np.ndarray): A 1d array with gradients with respect to each parameter value.

        Returns:
            np.ndarray: The new parameter values, the same shape as the initial parameter values.
        """

        pass

    @abstractmethod
    def is_converged(self: Self, grad: np.ndarray) -> bool:
        """
        Checks whether the convergence criteria of the operator is met.
        Method should be overridden to use gradient to chck convergence

        Args:
            self (Self): A reference to the current class instance.
            grad (np.ndarray): A 1d array with the gradients with respect to each parameter value.

        Returns:
            bool: Whether the optimizer has converged.
        """

        pass


class GradientFreeOptimizer(Optimizer):
    """
    Inherits from `Optimizer`. Base class for all optimizers not requiring gradients. Instead,
    calls to `update()` requires a function f that evaluates the function at specific parameter values.
    Calls to `is_converged()` are passed nothing and should rely on mutable optimizer state.
    """

    @abstractmethod
    def update(
        self: Self, param_vals: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """
        Performs a single step of optimization, returning the new param_vals,
        does not update param_vals in place. An abstract method in which subclasses
        should override with their implementation of a single update step.

        Args:
            self (Self): A reference to the current class instance.
            f (Callable[[np.ndarray], float]): A function that evaluates at different parameter values.

        Returns:
            np.ndarray: The new parameter values, the same shape as the initial parameter values.
        """

        pass

    @abstractmethod
    def is_converged(self: Self) -> bool:
        """
        Checks whether the convergence criteria of the operator is met.
        Method should be overriden using mutable state to determine convergence.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            bool: Whether the optimizer has converged.
        """

        pass


class HybridOptimizer(Optimizer):
    """
    Inherits from `Optimizer`. Base class for all optimizers requiring both gradients and a function that
    evaluates at different paramete values. Calls to `update()` requires both the gradient of the parameters
    and the function f. Calls to `is_converged()` are passing the gradient, and may determine convergence both
    through that and mutable state.
    """

    @abstractmethod
    def update(
        self: Self,
        param_vals: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """
        Performs a single step of optimization, returning the new param_vals,
        does not update param_vals in place. An abstract method in which subclasses
        should override with their implementation of a single update step.

        Args:
            self (Self): A reference to the current class instance.
            grad (np.ndarray): A 1d array with gradients with respect to each parameter value.
            f (Callable[[np.ndarray], float]): A function that evaluates at different parameter values.

        Returns:
            np.ndarray: The new parameter values, the same shape as the initial parameter values.
        """

        pass

    @abstractmethod
    def is_converged(self: Self, grad: np.ndarray) -> bool:
        """
        Checks whether the convergence criteria of the operator is met.
        Method should be overriden using gradient and / or mutable state to determine convergence.

        Args:
            self (Self): A reference to the current class instance.
            grad (np.ndarray): A 1d array with the gradients with respect to each parameter value.

        Returns:
            bool: Whether the optimizer has converged.
        """

        pass


class FunctionalOptimizer(Optimizer):
    """
    Inherits from `Optimizer`. An optimizer that wraps a functional interface, like `scipy.optimize.minimize`
    to perform optimization. This is clearly a work around for implementing several of the harder optimizers,
    like LBFGS, that do not work so well with our in place architecture.
    """

    @abstractmethod
    def update(
        self: Self,
        param_vals: np.ndarray,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        callback: Callable[[np.ndarray], None],
    ) -> None:
        """
        Performs the entire optimization process while calling callback each step of that process.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): The current parameter values.
            f (Callable[[np.ndarray], float]): A function f that maps from parameter values to function value.
            grad_f (Callable[[np.ndarray], np.ndarray]): A function grad_f that calculates the gradient of f at parameter values.
            callback (Callable[[np.ndarray], None]): A callback that takes the parameter values at each step.
        """

        pass
