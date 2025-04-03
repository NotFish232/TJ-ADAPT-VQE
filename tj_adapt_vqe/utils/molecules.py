from enum import Enum

from openfermion import MolecularData, QubitOperator
from openfermionpyscf import run_pyscf  # type: ignore
from qiskit.quantum_info.operators import SparsePauliOp  # type: ignore


class AvailableMolecules(Enum):
    H2 = "H2"


def make_molecule(m_type: AvailableMolecules, /, r: float) -> MolecularData:
    # TODO FIXME: make this function actually decent
    if m_type == AvailableMolecules.H2:
        geometry = [["H", [0, 0, 0]], ["H", [0, 0, r]]]
        basis = "sto-3g"
        multiplicity = 1
        charge = 0
        h2 = MolecularData(geometry, basis, multiplicity, charge, description="H2")
        h2 = run_pyscf(h2, run_fci=True, run_ccsd=True)

        return h2

    raise NotImplementedError()


def openfermion_to_qiskit(qubit_operator: QubitOperator, n_qubits: int) -> SparsePauliOp:
    """
    Converts from an opernfermion QubitOperator to a Qiskit SparsePauliOp

    Args:
        qubit_operator: QubitOperator, an openfermion QubitOperator
        n_qubits: int, the number of qubits the qubit operator is acting on
    """
    pauli_strs = []
    pauli_coeffs = []

    for q_op, coeff in qubit_operator.terms.items():
        s = ["I"] * n_qubits
        for i, p in q_op:
            s[i] = p

        pauli_strs.append("".join(s))
        pauli_coeffs.append(coeff)
    
    return SparsePauliOp(pauli_strs, pauli_coeffs)