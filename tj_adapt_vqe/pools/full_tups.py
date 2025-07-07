from qiskit.circuit import QuantumCircuit  # type: ignore
from qiskit.quantum_info.operators import SparsePauliOp  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Self, override

from ..ansatz.functional import make_tups_ansatz
from ..utils.molecules import Molecule
from .pool import Pool


class FullTUPSPool(Pool):
    """
    The full Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    pool of only a single operator, a single layer of the TUPS ansatz, so adapt vqe is just to find optimal number of layers
    """

    def __init__(self: Self, molecule: Molecule) -> None:
        super().__init__(molecule)

        self.n_qubits = molecule.data.n_qubits

        # this pool should only be used with the criterion checking for improvement
        # commutator of the identity is trivially 0
        self.operators = [SparsePauliOp("I" * self.n_qubits)]
        self.labels = ["L"]

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "full_tups_pool"

    @override
    def get_op(self: Self, idx: int) -> LinearOp:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def get_exp_op(self: Self, _: int) -> QuantumCircuit:
        return make_tups_ansatz(self.n_qubits, n_layers=1)

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
