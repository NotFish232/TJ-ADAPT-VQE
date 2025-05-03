from openfermion import MolecularData
from qiskit.circuit import Gate  # type: ignore
from qiskit.quantum_info.operators import SparsePauliOp  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Any, Self, override

from ..utils.ansatz import make_tups_ansatz
from .pool import Pool


class FullTUPSPool(Pool):
    """
    The full Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
    pool of only a single operator, a single layer of the TUPS ansatz, so adapt vqe is just to find optimal number of layers
    """

    def __init__(self: Self, molecule: MolecularData) -> None:
        super().__init__("Full TUPS Pool", molecule)

        self.n_spatials = molecule.n_qubits // 2

        # this pool should only be used with the criterion checking for improvement
        # commutator of the identity is trivially 0
        self.operators = [SparsePauliOp("I" * self.n_qubits)]
        self.labels = ["L"]

    @override
    def get_op(self: Self, idx: int) -> LinearOp:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def get_exp_op(self: Self, idx: int) -> Gate:
        return make_tups_ansatz(self.n_qubits, n_layers=1).to_gate(label="L")

    @override
    def to_config(self: Self) -> dict[str, Any]:
        return {"name": self.name, "n_spatials": self.n_spatials}

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)
