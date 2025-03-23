from qiskit import QuantumCircuit  # type: ignore
from qiskit.primitives import BackendEstimator  # type: ignore
from qiskit.quantum_info.operators.base_operator import BaseOperator  # type: ignore
from qiskit_aer import Aer  # type: ignore
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient  # type: ignore
from typing_extensions import Self

QISKIT_BACKEND = "qasm_simulator"



class Measure:
    """
    Calculates Gradients for each parameter via expectation values on an arbitrary qiskit backend

    params: QuantumCircuit, qiskit circuit to calculate gradients from
    paramters: list[float], current parameter values
    operator: BaseOperator, operator to calculate gradient wrt to
    num_shots: int, num_shots to run simulation for
    """

    def __init__(
        self: Self,
        circuit: QuantumCircuit,
        params: list[list[float]],
        operator: BaseOperator,
        num_shots: int = 4096,
    ) -> None:
        self.circuit = circuit
        self.params = params

        self.operator = operator
        self.num_shots = num_shots

        self.gradients = self._calculate_gradients()

    def _calculate_gradients(self: Self) -> None:
        backend = Aer.get_backend(QISKIT_BACKEND)

        estimator = BackendEstimator(backend=backend)
        estimator.options.default_precision = 1 / self.num_shots ** (1 / 2)

        gradient_estimator = ParamShiftEstimatorGradient(estimator)

        job_result = gradient_estimator.run(self.circuit, self.operator, self.params)

        return job_result.result().gradients
