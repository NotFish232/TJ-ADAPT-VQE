from openfermion import QubitOperator
from qiskit.quantum_info.operators import SparsePauliOp  # type: ignore


def openfermion_to_qiskit(qubit_operator: QubitOperator, n_qubits: int) -> SparsePauliOp:
    """
    Converts from an openfermion QubitOperator to a Qiskit SparsePauliOp

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