from openfermion import MolecularData
from qiskit import QuantumCircuit  # type: ignore
from typing_extensions import Self, override

from tj_adapt_vqe.pools import Pool
from tj_adapt_vqe.vqe import VQE


class ADAPTVQE(VQE):
    """
    Class implementing the ADAPT-VQE algorithm
    """

    def __init__(self: Self, molecule: MolecularData, pool: Pool) -> None:
        """
        Initializes the ADAPTVQE object
        Arguments:
            pool (Pool): the pool that the ADAPTVQE uses to form the Ansatz
        """
        self.pool = pool

    @override
    def make_ansatz(self: Self) -> QuantumCircuit:
        pass

    @override
    def run(self: Self):
        """
        Runs the ADAPTVQE algorithm. The main loop goes like this:
        1. measure gradient
        2. if converged, stop
        3. select operator with the largest gradient
        4. grow ansatz
        5. run self.optimize
        6. goto 1
        Returns some metrics of training, i.e. how the parameters are updating, what the current ground energy is etc
        """
