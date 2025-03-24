import numpy as np
from qiskit import QuantumCircuit  # type: ignore
from qiskit.primitives import BackendEstimator, BackendEstimatorV2  # type: ignore
from qiskit.quantum_info.operators.base_operator import BaseOperator  # type: ignore
from qiskit_aer import Aer  # type: ignore
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient  # type: ignore
from typing_extensions import Self

DEFAULT_QISKIT_BACKEND = "qasm_simulator"


class Measure:
    """
    Calculates Gradients and Expectation Values on a Qiskit Circuit
    Uses an Arbitrary Qiskit Backend along with a provided number of shots

    Args:
        circuit: QuantumCircuit, parameterized qiskit circuit that gradients are calculated on
        param_values: np.ndarray, current values of each parameter in circuit
        operator: BaseOperator, operator to calculate gradient wrt to
        qiskit_backend: str, backend to run qiskit on, defaults to DEFAULT_QISKIT_BACKEND
        num_shots: int, num_shots to run simulation for, defaults to 4096
    """

    def __init__(
        self: Self,
        circuit: QuantumCircuit,
        param_values: np.ndarray,
        operator: BaseOperator,
        qiskit_backend: str = DEFAULT_QISKIT_BACKEND,
        num_shots: int = 1024,
    ) -> None:
        self.circuit = circuit
        self.param_values = param_values

        self.operator = operator
        self.qiskit_backend = qiskit_backend
        self.num_shots = num_shots

        self.expectation_value = self._calculate_expectation_value()
        self.gradients = self._calculate_gradients()

    def _calculate_expectation_value(self: Self) -> float:
        """
        Calculates and returns the expectation value of the operator using the quantum circuit
        """
        backend = Aer.get_backend(self.qiskit_backend)

        estimator = BackendEstimatorV2(backend=backend)
        estimator.options.default_precision = 1 / self.num_shots ** (1 / 2)

        job_result = estimator.run([(self.circuit, self.operator, self.param_values)])

        return job_result.result()[0].data.evs

    def _calculate_gradients(self: Self) -> np.ndarray:
        """
        Calculates and returns a numpy float32 array representing the gradient of each parameter
        """

        backend = Aer.get_backend(self.qiskit_backend)

        estimator = BackendEstimator(backend=backend)
        estimator.options.default_precision = 1 / self.num_shots ** (1 / 2)

        gradient_estimator = ParamShiftEstimatorGradient(estimator)

        job_result = gradient_estimator.run(
            self.circuit, self.operator, [self.param_values]
        )

        return job_result.result().gradients[0]
