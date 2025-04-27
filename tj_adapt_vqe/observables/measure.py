import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit  # type: ignore
from qiskit.primitives import EstimatorResult  # type: ignore
from qiskit.primitives.backend_estimator import Options  # type: ignore
from qiskit.quantum_info import Statevector  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore
from qiskit_aer.primitives import EstimatorV2 as Estimator  # type: ignore
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient  # type: ignore
from typing_extensions import Any, Self

from .observable import Observable

DEFAULT_BACKEND = AerSimulator(method="automatic", device="GPU")


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

    Args:
        circuit: QuantumCircuit, parameterized qiskit circuit that gradients are calculated on
        param_values: np.ndarray, current values of each parameter in circuit
        ev_observables: list[Observable], observables to calculate expectation values against,
        grad_observables: list[Observable], observables to calcualte gradients wrt to
        qiskit_backend: str, backend to run qiskit on, defaults to DEFAULT_QISKIT_BACKEND
        num_shots: int, num_shots to run simulation for, defaults to 1024

    """

    def __init__(
        self: Self,
        circuit: QuantumCircuit,
        param_values: np.ndarray,
        ev_observables: list[Observable] = [],
        grad_observables: list[Observable] = [],
        num_shots: int = 1024,
    ) -> None:
        self.circuit = circuit
        self.param_values = param_values

        self.ev_observables = ev_observables
        self.grad_observables = grad_observables

        self.simulator = DEFAULT_BACKEND

        self.num_shots = num_shots

        # estimator used for both expectation value and gradient calculations
        self.estimator = Estimator.from_backend(self.simulator)
        self.estimator.options.default_precision = 1 / self.num_shots ** (1 / 2)

        # initialize ParamShiftEstimatorGradient by wrapper the estimator class
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

        job_result = self.estimator.run(
            [
                (self.circuit, obv.operator, self.param_values)
                for obv in self.ev_observables
            ]
        ).result()

        return {
            obv: jr.data.evs.item() for obv, jr in zip(self.ev_observables, job_result)
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
            [self.param_values] * len(self.grad_observables),
        ).result()

        return {obv: jr for obv, jr in zip(self.grad_observables, job_result.gradients)}


def exact_expectation_value(circuit: QuantumCircuit, operator: ArrayLike) -> float:
    """
    Calculates the exact expectation value of a state prepared by a qiskit quantum circuit using statevector evolution
    Notes: assumes the operator is Hermitian and thus has a real expectation value. Returns the real component of whatever is calculated

    Args:
        circuit: QuantumCircuit, the circuit object that an empty state should be evolved from,
        operator: ArrayLike, an array like object that can be used to calculate expection value,
    """
    statevector = Statevector.from_label("0" * circuit.num_qubits)

    statevector = statevector.evolve(circuit)

    state_array = statevector.data

    return (state_array.conjugate().transpose() @ operator @ state_array).real
