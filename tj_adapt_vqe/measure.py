from typing_extensions import Self

from qiskit import QuantumCircuit  # type: ignore


class Measure:
    def __init__(self: Self):
        pass

    def setup(self: Self, ansatz: QuantumCircuit):
        pass