from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from typing_extensions import Any, Callable, Self


class OptimizerType(str, Enum):
    """
    Inherits from `enum.Enum`. An enum representing the different types of available optimizers.

    Members:
        `OptimizerType.Gradient` represents an optimizer that requires the gradient of each parameter.
        `OptimizerType.NonGradient` represents an optimizer that requires a callable that maps parameters => function value.
        `OptimizerType.Hybrid` represents an optimizer that requires both a gradient and the callable.
        `OptimizerType.Functional` represents an optimizer that wraps a functional interface.

    """

    Gradient = "Gradient"
    NonGradient = "NonGradient"
    Hybrid = "Hybrid"
    Functional = "Functional"


class Optimizer(ABC):
    """
    Inherits from `abc.ABC`. A base class that each other optimizer should inherit from.
    Subclasses should define the methods `update(...)` and `is_converged(...)` with different arguments.
    """

    def __init__(self: Self, name: str, type: OptimizerType) -> None:
        """
        Constructs an instance of an Optimizer.

        Args:
            self (Self): A reference to the current class instance.
            name (str): The name of the optimizer, used for configs and logging.
            type (OptimizerType): The type of the optimizer: Gradient, NonGradient, or Hyrbid.
        """

        self.name = name
        self.type = type

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

    def to_config(self: Self) -> dict[str, Any]:
        """
        Converts the optimizer state to a configuration that can then be used to recreate the optimizer.
        This method should return each key value pair necessary to uniquely define the current optimizer,
        omitting any mutable state needed. Subclasses should override this method, returning their unique
        configuration options unioned by the output of `super().to_config()`.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            dict[str,Any]: A dictionary representation of the config where values can be anything.
        """

        return {"name": self.name, "type": self.type}

    def __str__(self: Self) -> str:
        """
        An implemention of the dunder method `__str__()`. Converts the `Optimizer` instance into
        a printable string. Produces the format Name(k1=v1,k2=v2) for each key and value in config.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            str: The printable string representation of the current optimizer.
        """

        config = self.to_config()
        return f"{self.name}({','.join(f'{k}={v}' for k, v in config.items())})"

    def __repr__(self: Self) -> str:
        """
        An implementation of the dunder method `__repr__()`. Converts the `Optimizer` to a string
        by calling `repr(...)` on the returned string from calling `str(...)` on self.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            str: The representation of the current optimizer.
        """

        return repr(str(self))


class GradientOptimizer(Optimizer):
    """
    Inherits from `Optimizer`. Base class for all optimizers requiring gradients for calls to `update()`
    and `is_converged()`.
    """

    def __init__(self: Self, name: str, grad_conv_threshold: float = 0.01) -> None:
        """
        Constructs an instance of a GradientOptimizer. Calls `super().__init()` with the passed name argument
        and a type of `OptimizerType.Gradient`.

        Args:
            self (Self): A reference to the current class instance.
            name (str): The name of the current optimizer.
            grad_conv_threshold (float, optional): The threshold for gradients to determine convergence. Defaults to 0.01.
        """

        super().__init__(name, OptimizerType.Gradient)

        self.grad_conv_threshold = grad_conv_threshold

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

    def is_converged(self: Self, grad: np.ndarray) -> bool:
        """
        Checks whether the convergence criteria of the operator is met. Base implementation is
        checking whether the maximum absolute value of the gradient is less than `self.grad_conv_threshold`.
        Method should be overriden if subclasses require a different convergence criteria.

        Args:
            self (Self): A reference to the current class instance.
            grad (np.ndarray): A 1d array with the gradients with respect to each parameter value.

        Returns:
            bool: Whether the optimizer has converged.
        """

        return np.max(np.abs(grad)) < self.grad_conv_threshold


class NonGradientOptimizer(Optimizer):
    """
    Inherits from `Optimizer`. Base class for all optimizers not requiring gradients. Instead,
    calls to `update()` requires a function f that evaluates the function at specific parameter values.
    Calls to `is_converged()` are passed nothing and should rely on mutable optimizer state.
    """

    def __init__(self: Self, name: str) -> None:
        """
        Constructs an instance of a NonGradientOptimizer. Calls `super().__init()` with the passed name argument
        and a type of `OptimizerType.NonGradient`.

        Args:
            self (Self): A reference to the current class instance.
            name (str): The name of the current optimizer.
        """

        super().__init__(name, OptimizerType.NonGradient)

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

    def __init__(self: Self, name: str) -> None:
        """
        Constructs an instance of a HybridOptimizer. Calls `super().__init()` with the passed name argument
        and a type of `OptimizerType.Hybrid`.

        Args:
            self (Self): A reference to the current class instance.
            name (str): The name of the current optimizer.
        """

        super().__init__(name, OptimizerType.Hybrid)

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

    def __init__(self: Self, name: str) -> None:
        """
        Constructs an instance of a FunctionalOptimizer. Calls `super().__init()` with the passed name argument
        and a type of `OptimizerType.Functional`.

        Args:
            self (Self): A reference to the current class instance.
            name (str): The name of the current optimizer.
        """

        super().__init__(name, OptimizerType.Functional)

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
