# from openfermion import MolecularData, jordan_wigner
# from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
# from typing_extensions import Any, Self, override

# from .pool import Pool


# class FullTUPSPool(Pool):
#     """
#     The full Tiled Unitary Product State pool, which uses operators from https://arxiv.org/pdf/2312.09761
#     pool of only a single operator, so adapt vqe is just to find optimal number of layers
#     """

#     def __init__(self: Self, molecule: MolecularData) -> None:
#         super().__init__("Full TUPS Pool", molecule)

#         self.n_spatials = molecule.n_qubits // 2

#         # self.operators, self.labels = self.make_operators_and_labels()

#     # def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
#     #     tups_operator = 

#     @override
#     def to_config(self: Self) -> dict[str, Any]:
#         return {"name": self.name, "n_spatials": self.n_spatials}
