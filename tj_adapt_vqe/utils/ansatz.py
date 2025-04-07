from openfermion import FermionOperator, jordan_wigner, normal_ordered
from qiskit.circuit import (  # type: ignore
    Gate,
    Parameter,
    QuantumCircuit,
)
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore

from .molecules import openfermion_to_qiskit


def normalize_op(operator: FermionOperator) -> FermionOperator:
    """
    Normalizes a symbolic operator by making the magnitudes of the coefficients sum to 0
    """

    return operator / sum(abs(c) for c in operator.terms.values())


def create_one_body_op(p: int, q: int) -> FermionOperator:
    """
    Returns a generalized one body fermionic operator acting on spatial orbitals p and q
    """
    e_pq = FermionOperator(f"{2 * p}^ {2 * q}") + FermionOperator(
        f"{2 * p + 1}^ {2 * q + 1}"
    )
    e_qp = FermionOperator(f"{2 * q}^ {2 * p}") + FermionOperator(
        f"{2 * q + 1}^ {2 * p + 1}"
    )

    op = e_pq - e_qp

    return normalize_op(normal_ordered(op))


def create_two_body_op(p: int, q: int) -> FermionOperator:
    """
    Returns a generalized two body fermionic operator acting on spacial orbitals p and q
    """
    e_pq = FermionOperator(f"{2 * p}^ {2 * q}") + FermionOperator(
        f"{2 * p + 1}^ {2 * q + 1}"
    )
    e_qp = FermionOperator(f"{2 * q}^ {2 * p}") + FermionOperator(
        f"{2 * q + 1}^ {2 * p + 1}"
    )

    op = e_pq**2 - e_qp**2

    return normalize_op(normal_ordered(op))


def create_parameterized_unitary_op(
    p: int, q: int, layer: int
) -> tuple[Gate, Gate, Gate]:
    """
    Creates a unitary operator that is parameterized by 3 operators and is acting on
    Spacial orbitals p and q

    Args:
        p: int, first orbital to act on,
        q: int, second orbital to act on,
        layer: int, which layer unitary operator is on (only used for parameter naming)
    """

    # hard code the orbitals it maps to
    # orbitals will be mapped correctly when converting it to qiskit
    one_body_op = create_one_body_op(0, 1)
    two_body_op = create_two_body_op(0, 1)

    # apply the jordan wigner transformation and make operators strictly real
    # since qiskit PauliEvolution adds the i to the exponentiation
    # similarly * -1 to counteract the PauliEvolutionGate
    # i * -i = 1
    one_body_op_jw = jordan_wigner(1j * one_body_op)
    two_body_op_jw = jordan_wigner(1j * two_body_op)

    # convert the jw representations to a qiskit compatible format (SparsePauliOp)
    one_body_op_qiskit = openfermion_to_qiskit(one_body_op_jw, 4)
    two_body_op_qiskit = openfermion_to_qiskit(two_body_op_jw, 4)

    params = [Parameter(f"l{layer}p{p}q{q}θ{i + 1}") for i in range(3)]

    qc = QuantumCircuit(4)

    gate_1 = PauliEvolutionGate(one_body_op_qiskit, params[0])
    gate_2 = PauliEvolutionGate(two_body_op_qiskit, params[1])
    gate_3 = PauliEvolutionGate(one_body_op_qiskit, params[2])

    qc.append(gate_3, range(4))
    qc.append(gate_2, range(4))
    qc.append(gate_1, range(4))

    return qc.to_gate(label="U")


def make_tups_ansatz(n_qubits: int, n_layers: int = 5) -> QuantumCircuit:
    """
    Implements the Tiled Unitary Process State Ansatz for a molecule from this paper: https://arxiv.org/pdf/2312.09761

    Args:
        n_qubits: int, the number of qubits the circuit should take in,
        n_layers: int, the number of layers to repeat the TUPS ansatz, defaults to 5
    """

    qc = QuantumCircuit(n_qubits)

    L = n_layers

    N = n_qubits // 2
    A = (N - 1) // 2
    B = N // 2

    # initialize spatial orbitals in perfect pairing
    for i in range(n_qubits):
        if i // 2 % 2 == 0:
            qc.x(i)

    for l in range(1, L + 1):
        for p in range(1, B + 1):
            u = create_parameterized_unitary_op(2 * p, 2 * p - 1, l)
            qc.append(u, range(4 * (p - 1), 4 * p))
        for p in range(1, A + 1):
            u = create_parameterized_unitary_op(2 * p + 1, 2 * p, l)
            qc.append(u, range(2 + 4 * (p - 1), 2 + 4 * p))

    return qc
