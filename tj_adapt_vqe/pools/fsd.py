from itertools import combinations
from typing_extensions import Self, override

from .pool import Pool
from ..utils import openfermion_to_qiskit

from openfermion import FermionOperator, MolecularData, jordan_wigner, normal_ordered
from qiskit.quantum_info.operators import SparsePauliOp


class FSD(Pool):
    """
    The fermionic single and double excitations pool.
    """

    def __init__(self: Self, molecule: MolecularData, n_excitations: int) -> None:
        self.n_excitations = n_excitations
        super().__init__(molecule)


    @override
    def make_operators(self: Self) -> None:
        pool = []
        labels = []
        for n in range(1, self.n_excitations+1):
            occupied = [*combinations(range(0, self.n_electrons), n)]
            virtual = [*combinations(range(self.n_electrons, self.n_qubits), n)]
            ops = [openfermion_to_qiskit(jordan_wigner(FermionOperator(' '.join(f'{j}^' for j in v) + ' ' + ' '.join(str(j) for j in o))), self.n_qubits) for o in occupied for v in virtual]
            ops = [1j*(op - op.conjugate().transpose()).simplify() for op in ops]
            pool += ops
            labels += [f'o{o}v{v}' for v in virtual for o in occupied]

        self.operators = pool
        self.labels = labels