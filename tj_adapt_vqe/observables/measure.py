import numpy as np
from qiskit import QuantumCircuit  # type: ignore
from qiskit.primitives import EstimatorResult  # type: ignore
from qiskit.primitives.backend_estimator import Options  # type: ignore
from qiskit.quantum_info import Statevector  # type: ignore
from qiskit_aer.primitives import EstimatorV2 as Estimator  # type: ignore
from qiskit_algorithms.gradients import (  # type: ignore
    FiniteDiffEstimatorGradient,
    ParamShiftEstimatorGradient,
)
from typing_extensions import Any, Callable, Self

from .observable import Observable
from .qiskit_backend import QiskitBackend


class EstimatorResultWrapper:
    """
    Wraps an EstimatorResult object and returns the actual contained valued on the .result() call
    """

    def __init__(self: Self, estimator_result: EstimatorResult) -> None:
        self.estimator_result = estimator_result

    def result(self: Self) -> EstimatorResult:
        return self.estimator_result


class GradientCompatibleEstimatorV2:
    """
    Wraps a BackendEstimatorV2 instance and makes it compatible with all the param shift estimator classes
    (which still use the interface from BackendEstimatorV1)
    """

    def __init__(self: Self, estimator: Estimator) -> None:
        self.estimator = estimator

    @property
    def options(self: Self) -> Options:
        return Options()

    def run(self: Self, *args: tuple[Any], **kwargs: tuple[str, Any]) -> Any:
        t_args = [*zip(*args)]

        job_result = self.estimator.run(t_args, **kwargs).result()
        values = np.array([x.data.evs.item() for x in job_result])
        metadata = [x.metadata for x in job_result]

        return EstimatorResultWrapper(EstimatorResult(values, metadata))


class Measure:
    """
    Calculates Gradients and Expectation Values on a Qiskit Circuit
    Uses an Arbitrary Qiskit Backend along with a provided number of shots
    """

    def __init__(
        self: Self,
        circuit: QuantumCircuit,
        param_vals: np.ndarray,
        ev_observables: list[Observable] = [],
        grad_observables: list[Observable] = [],
        qiskit_backend: QiskitBackend = QiskitBackend.Exact(),
    ) -> None:
        """
        Instantiates a `Measure` class instance.

        Args:
            circuit (QuantumCircuit): parameterized qiskit circuit that gradients are calculated on.
            param_values (np.ndarray): current values of each parameter in circuit.
            ev_observables (list[Observable], optional): observables to calculate expectation values against. Defaults to [].
            grad_observables (list[Observable], optional): observables to calcualte gradients wrt to. Defaults to [].
            qiskit_backend (QiskitBackend, optional): backend to run qiskit on. Defaults to `QiskitBackend.Exact()`.
        """

        self.circuit = circuit
        self.param_vals = param_vals

        self.ev_observables = ev_observables
        self.grad_observables = grad_observables

        # estimator used for both expectation value and gradient calculations
        self.qiskit_backend = qiskit_backend
        self.estimator = Estimator.from_backend(self.qiskit_backend.data)

        # apply shot noise
        num_shots = self.qiskit_backend.data.options.shots
        if num_shots != 0:
            self.estimator.options.default_precision = 1 / num_shots ** (1 / 2)

        # initialize ParamShiftEstimatorGradient by wrapper the estimator class
        # Finite Diff Estimator is only remotely usable for noiseless simulations
        if num_shots == 0:
            self.gradient_estimator = FiniteDiffEstimatorGradient(
                GradientCompatibleEstimatorV2(self.estimator), 1e-8
            )
        else:
            self.gradient_estimator = ParamShiftEstimatorGradient(
                GradientCompatibleEstimatorV2(self.estimator)
            )

        self.evs = self._calculate_expectation_value()
        self.grads = self._calculate_gradients()

    def _calculate_expectation_value(self: Self) -> dict[Observable, float]:
        """
        Calculates and returns the expectation value of the operator using the quantum circuit
        """

        if len(self.ev_observables) == 0:
            return {}

        # for exact simulation use statevectors
        if self.qiskit_backend.data.options.shots == 0:
            statevector = Statevector.from_label("0" * self.circuit.num_qubits)
            statevector = statevector.evolve(
                self.circuit.assign_parameters(
                    {p: v for p, v in zip(self.circuit.parameters, self.param_vals)}
                )
            )

            vec = statevector.data

            return {
                o: (vec.conjugate().transpose() @ o.operator_matrix @ vec).item().real
                for o in self.ev_observables
            }
        else:
            job_result = self.estimator.run(
                [
                    (self.circuit, obv.operator, self.param_vals)
                    for obv in self.ev_observables
                ]
            ).result()

            return {
                obv: jr.data.evs.item()
                for obv, jr in zip(self.ev_observables, job_result)
            }

    def _calculate_gradients(self: Self) -> dict[Observable, np.ndarray]:
        """
        Calculates and returns a numpy float32 array representing the gradient of each parameter
        """

        if len(self.grad_observables) == 0:
            return {}

        job_result = self.gradient_estimator.run(
            [self.circuit] * len(self.grad_observables),
            [obv.operator for obv in self.grad_observables],
            [self.param_vals] * len(self.grad_observables),
        ).result()

        return {obv: jr for obv, jr in zip(self.grad_observables, job_result.gradients)}


def make_ev_function(
    circuit: QuantumCircuit,
    observable: Observable,
    qiskit_backend: QiskitBackend,
) -> Callable[[np.ndarray], float]:
    """
    Makes a function that evaluates the circuit at different parameter values
    and returns the expectation value of the observable. Used for optimizers that
    require the actual function values to perform optimization.

    Args:
        circuit (QuantumCircuit): The parameterized quantum circuit that observables are calculated on.
        observable (Observable): The observable to calculate the expectation value of.
        qiskit_backend (QiskitBackend): The qiskit backend to run simulations on.

    Returns:
        Callable[[np.ndarray],float]: A callable that returns the expectation value.
    """

    def _ev_function(param_vals: np.ndarray) -> float:
        m = Measure(
            circuit,
            param_vals,
            ev_observables=[observable],
            qiskit_backend=qiskit_backend,
        )

        return m.evs[observable]

    return _ev_function


def make_grad_function(
    circuit: QuantumCircuit, observable: Observable, qiskit_backend: QiskitBackend
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Makes a function that evaluates the circuit at different parameter values
    and returns the gradient of the observable. Used for optimizers that
    require the gradient function, like scipy optimizers.

    Args:
        circuit (QuantumCircuit): The parameterized quantum circuit that observables are calculated on.
        observable (Observable): The observable to calculate the expectation value of.
        qiskit_backend (QiskitBackend): The qiskit backend to run simulations on.

    Returns:
        Callable[[np.ndarray],np.ndarray]: A callable that returns the gradient.
    """

    def _ev_function(param_vals: np.ndarray) -> np.ndarray:
        m = Measure(
            circuit,
            param_vals,
            grad_observables=[observable],
            qiskit_backend=qiskit_backend,
        )

        return m.grads[observable]

    return _ev_function
