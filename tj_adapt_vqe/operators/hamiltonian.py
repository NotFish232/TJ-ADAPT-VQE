from typing_extensions import Self

from tj_adapt_vqe.operators.observable import Observable


class Hamiltonian(Observable):
    def evaluate(self: Self, bitstring: str):
        pass