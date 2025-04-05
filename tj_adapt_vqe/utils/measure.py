import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit  # type: ignore
from qiskit.primitives import BackendEstimatorV2, EstimatorResult  # type: ignore
from qiskit.primitives.backend_estimator import Options  # type: ignore
from qiskit.quantum_info import Statevector  # type: ignore
from qiskit.quantum_info.operators.base_operator import BaseOperator  # type: ignore
from qiskit_aer import Aer  # type: ignore
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient  # type: ignore
from typing_extensions import Any, Self

DEFAULT_QISKIT_BACKEND = "statevector_simulator"


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

    def __init__(self: Self, estimator_v2: BackendEstimatorV2) -> None:
        self.estimator_v2 = estimator_v2

    @property
    def options(self: Self) -> Options:
        return Options()

    def run(self: Self, *args: tuple[Any], **kwargs: tuple[str, Any]) -> Any:
        t_args = [*zip(*args)]
        job_result = self.estimator_v2.run(t_args, **kwargs).result()

        values = np.array([x.data.evs.item() for x in job_result])

        metadata = [
            {"variance": x.data.stds, "shots": x.metadata["shots"]} for x in job_result
        ]

        return EstimatorResultWrapper(EstimatorResult(values, metadata))


class Measure:
    """
    Calculates Gradients and Expectation Values on a Qiskit Circuit
    Uses an Arbitrary Qiskit Backend along with a provided number of shots

    Args:
        circuit: QuantumCircuit, parameterized qiskit circuit that gradients are calculated on
        param_values: np.ndarray, current values of each parameter in circuit
        operator: BaseOperator, operator to calculate gradient wrt to
        qiskit_backend: str, backend to run qiskit on, defaults to DEFAULT_QISKIT_BACKEND
        num_shots: int, num_shots to run simulation for, defaults to 1024
        should_calculate_expectation_values: bool, whether to compute expectation values
        should_calculate_gradients: bool, whether to compute gradients
    """


    def __init__(
        self: Self,
        circuit: QuantumCircuit,
        param_values: np.ndarray,
        operator: BaseOperator,
        qiskit_backend: str = DEFAULT_QISKIT_BACKEND,
        num_shots: int = 1024,
        should_calculate_expectation_values: bool = True,
        should_calculate_gradients: bool = True,
    ) -> None:
        self.circuit = circuit
        self.param_values = param_values

        self.operator = operator
        self.num_shots = num_shots

        self.should_calculate_expectation_values = should_calculate_expectation_values
        self.should_calculate_gradients = should_calculate_gradients

        self.backend = Aer.get_backend(qiskit_backend)

        # estimator used for both expectation value and gradient calculations
        self.estimator = BackendEstimatorV2(backend=self.backend)
        self.estimator.options.default_precision = 1 / self.num_shots ** (1 / 2)

        # initialize ParamShiftEstimatorGradient by wrapper the estimator class
        self.gradient_estimator = ParamShiftEstimatorGradient(
            GradientCompatibleEstimatorV2(self.estimator)
        )

        if self.should_calculate_expectation_values:
            self.expectation_value = self._calculate_expectation_value()
        if self.should_calculate_gradients:
            self.gradients = self._calculate_gradients()

    def _calculate_expectation_value(self: Self) -> float:
        """
        Calculates and returns the expectation value of the operator using the quantum circuit
        """
        job_result = self.estimator.run(
            [(self.circuit, self.operator, self.param_values)]
        )

        # print(job_result.result())

        return job_result.result()[0].data.evs

    def _calculate_gradients(self: Self) -> np.ndarray:
        """
        Calculates and returns a numpy float32 array representing the gradient of each parameter
        """
        job_result = self.gradient_estimator.run(
            self.circuit, self.operator, [self.param_values]
        )

        # print(job_result.result())

        return job_result.result().gradients[0]


def exact_expectation_value(circuit: QuantumCircuit, operator: ArrayLike) -> float:
    """
    Calculates the exact expectation value of a state prepared by a qiskit quantum circuit using statevector evolution
    Notes: assumes the operator is Hermetian and thus has a real expectation value. Returns the real component of whatever is calcualted

    Args:
        circuit: QuantumCircuit, the circuit object that an empty state should be evolved from,
        operator: ArrayLike, an array like object that can be used to calculate expection value,
    """
    statevector = Statevector.from_label("0" * circuit.num_qubits)

    statevector = statevector.evolve(circuit)

    state_array = statevector.data

    return (state_array.conjugate().transpose() @ operator @ state_array).real
