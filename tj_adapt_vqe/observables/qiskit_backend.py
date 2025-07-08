from qiskit_aer import AerSimulator  # type: ignore
from qiskit_aer.noise import NoiseModel  # type: ignore
from qiskit_ibm_runtime.fake_provider import FakeVigoV2  # type: ignore
from typing_extensions import Protocol, Self, Type, override

from ..utils.serializable import Serializable


class QiskitBackendConstructor(Protocol):
    def __call__(self: Self) -> "QiskitBackend": ...


class QiskitBackend(Serializable):
    """
    Inherits from `Serializable`. A wrapper around qiskit backend similar to `Molecule`.
    """

    Exact: QiskitBackendConstructor
    ShotNoise: QiskitBackendConstructor
    HardwareNoise: QiskitBackendConstructor

    _all: list[QiskitBackendConstructor] = []

    def __init__(
        self: Self,
        shots: int = 2**20,
        method: str = "automatic",
        device: str = "CPU",
        use_noise_model: bool = False,
    ) -> None:
        """

        Args:
            self (Self): A reference to the current class instance.
            shots (int, optional): Number of shots, 0 is noiseless. Defaults to 2**20.
            method (str, optional): Which method for simulation. Defaults to "automatic".
            device (str, optional): Which device to run on. Defaults to "CPU".
            use_noise_model (bool, optional): Whether to use a noise model. Defaults to False.
        """

        self.shots = shots
        self.method = method
        self.device = device
        self.use_noise_model = use_noise_model

        noise_model = None
        if use_noise_model:
            noise_model = NoiseModel.from_backend(FakeVigoV2())

        self.data = AerSimulator(
            shots=shots,
            method=method,
            noise_model=noise_model,
            device=device,
        )

    @staticmethod
    @override
    def _type() -> str:
        """
        The type of the class. Used in `Serializable`.
        """

        return "qiskit_backend"

    @staticmethod
    @override
    def _name() -> str:
        """
        Not needed. Like `Molecule`.
        """

        return ""

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        The params that the config has. Used in `Serializable`.
        """

        return ["shots", "method", "device", "use_noise_model"]

    @classmethod
    @override
    def all(cls: Type["QiskitBackend"]) -> list[Type["QiskitBackend"]]:
        raise NotImplementedError(
            "For non-abstract classes, use all_constructors instead()."
        )

    @classmethod
    @override
    def all_constructors(cls: Type["QiskitBackend"]) -> list[QiskitBackendConstructor]:
        """
        Little bit suspect
        """

        return QiskitBackend._all


def _qiskit_backend_constructor_wrapper(
    shots: int = 2**20,
    method: str = "automatic",
    device: str = "CPU",
    use_noise_model: bool = False,
) -> QiskitBackendConstructor:
    """
    Wraps the constructor of the QiskitBackend class.

    Returns:
        QiskitBackendConstructor: A new constructor that only takes a radius and run_fci.
    """

    def _callable() -> QiskitBackend:
        return QiskitBackend(shots, method, device, use_noise_model)

    QiskitBackend._all.append(_callable)

    return _callable


QiskitBackend.Exact = _qiskit_backend_constructor_wrapper(shots=0)
QiskitBackend.ShotNoise = _qiskit_backend_constructor_wrapper()
QiskitBackend.HardwareNoise = _qiskit_backend_constructor_wrapper(use_noise_model=True)
