from abc import ABC, abstractmethod
from itertools import combinations

from openfermion import FermionOperator, MolecularData, jordan_wigner, normal_ordered
from qiskit.circuit import Parameter, QuantumCircuit  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore
from typing_extensions import Self, override

from .conversions import openfermion_to_qiskit, prepend_params


class Ansatz(ABC):
    """
    Inherits from `abc.ABC`. Base class for all other Ansatzes. Completely
    unncessary but just for a nice interface (the lengths I would go for
    rust style enums in python).
    """

    def __init__(self: Self, name: str) -> None:
        """
        Constructs an Ansatz instance.

        Args:
            self (Self): A reference to the current class instance.
            name (str): The name of the Ansatz.
        """

        self.name = name

    @abstractmethod
    def construct(self: Self, molecule: MolecularData) -> QuantumCircuit:
        """
        Constructs an instance of the associated ansastz given a molecule.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The quantum circuit associated with the ansatz and the molecule.
        """

        pass

    def __str__(self: Self) -> str:
        """
        An implementation of the dunder method `__str__`.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            str: The string repsentation of the optimizer.
        """

        return self.name

    def __repr__(self: Self) -> str:
        """
        An implementation of the dunder method `__repr__`.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            str: The representation of the optimizer.
        """

        return repr(str(self))


class HartreeFockAnsatz(Ansatz):
    """
    Inherits from `Ansatz`. The HartreeFock ansatz.
    """

    def __init__(self: Self) -> None:
        """
        Constructs an instance of a HartreeFock.

        Args:
            self (Self): A reference to the current class instance.
        """

        super().__init__("hartree_fock_ansatz")

    @override
    def construct(self: Self, molecule: MolecularData) -> QuantumCircuit:
        """
        Generates the hartree fock ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The hartree fock quantum circuit.
        """

        return make_hartree_fock_ansatz(molecule.n_qubits, molecule.n_electrons)


class PerfectPairAnsatz(Ansatz):
    """
    Inherits from `Ansatz`. The PerfectPair ansatz.
    """

    def __init__(self: Self) -> None:
        """
        Constructs an instance of a PerfectPairAnsatz.

        Args:
            self (Self): A reference to the current class instance.
        """

        super().__init__("perfect_pair_ansatz")

    @override
    def construct(self: Self, molecule: MolecularData) -> QuantumCircuit:
        """
        Generates the perfect pair ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The perfect pairing quantum circuit.
        """

        return make_perfect_pair_ansatz(molecule.n_qubits)


class TUPSAnsatz(Ansatz):
    """
    Inherits from `Ansatz`. The TUPS (Tiled Unitary Product State) ansatz.
    """

    def __init__(self: Self, n_layers: int) -> None:
        """
        Constructs an instance of a TUPSAnsatz.

        Args:
            self (Self): A reference to the current class instance.
            num_layers (int): The number of layers to tile the ansatz for.
        """

        super().__init__("tups_ansatz")

        self.n_layers = n_layers

    @override
    def construct(self: Self, molecule: MolecularData) -> QuantumCircuit:
        """
        Generates the tups ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The TUPS quantum circuit.
        """

        return make_tups_ansatz(molecule.n_qubits, self.n_layers)


class UCCAnsatz(Ansatz):
    """
    Inherits from `Ansatz`. The UCC ansatz.
    """

    def __init__(self: Self, n_excitations: int) -> None:
        """
        Constructs an instance of a UCCAnsatz.

        Args:
            self (Self): A reference to the current class instance.
            n_excitations (int): The number of excitations for the UCC ansatz.
        """

        super().__init__("ucc_ansatz")

        self.n_excitations = n_excitations

    @override
    def construct(self: Self, molecule: MolecularData) -> QuantumCircuit:
        """
        Generates the UCC ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The UCC quantum circuit.
        """

        return make_ucc_ansatz(
            molecule.n_qubits, molecule.n_electrons, self.n_excitations
        )


def make_hartree_fock_ansatz(n_qubits: int, n_electrons: int) -> QuantumCircuit:
    """
    Creates an initial ansatz in the hartree fock state
    """

    qc = QuantumCircuit(n_qubits)

    for i in range(n_electrons):
        qc.x(n_qubits - i - 1)

    return qc


def make_perfect_pair_ansatz(n_qubits: int) -> QuantumCircuit:
    """
    Creates an initial ansatz in the perfect pairing configuration
    """

    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        if i // 2 % 2 == 0:
            qc.x(i)

    return qc


def normalize_op(operator: FermionOperator) -> FermionOperator:
    """
    Normalizes a symbolic operator by making the magnitudes of the coefficients sum to 0
    """

    return operator / sum(abs(c) for c in operator.terms.values())


def make_one_body_op(p: int, q: int) -> FermionOperator:
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


def make_two_body_op(p: int, q: int) -> FermionOperator:
    """
    Returns a generalized two body fermionic operator acting on spatal orbitals p and q
    """
    e_pq = FermionOperator(f"{2 * p}^ {2 * q}") + FermionOperator(
        f"{2 * p + 1}^ {2 * q + 1}"
    )
    e_qp = FermionOperator(f"{2 * q}^ {2 * p}") + FermionOperator(
        f"{2 * q + 1}^ {2 * p + 1}"
    )

    op = e_pq**2 - e_qp**2

    return normalize_op(normal_ordered(op))


def make_parameterized_unitary_op(p: int, q: int) -> QuantumCircuit:
    """
    Creates a unitary operator that is parameterized by 3 operators and is acting on
    spatial orbitals p and q

    Args:
        p: int, first orbital to act on,
        q: int, second orbital to act on,
        layer: int, which layer unitary operator is on (only used for parameter naming)
    """

    # hard code the orbitals it maps to
    # orbitals will be mapped correctly when converting it to qiskit
    one_body_op = make_one_body_op(0, 1)
    two_body_op = make_two_body_op(0, 1)

    # apply the jordan wigner transformation and make operators strictly real
    one_body_op_jw = jordan_wigner(one_body_op)
    two_body_op_jw = jordan_wigner(two_body_op)

    # convert the jw representations to a qiskit compatible format (SparsePauliOp)
    one_body_op_qiskit = openfermion_to_qiskit(one_body_op_jw, 4)
    two_body_op_qiskit = openfermion_to_qiskit(two_body_op_jw, 4)

    params = [Parameter(f"p{p}q{q}θ{i + 1}") for i in range(3)]

    qc = QuantumCircuit(4)

    # since qiskit PauliEvolutionGate adds the i to the exponentiation
    # similarly * -1 to counteract the PauliEvolutionGate
    # i * -i = 1
    gate_1 = PauliEvolutionGate(1j * one_body_op_qiskit, params[0])
    gate_2 = PauliEvolutionGate(1j * two_body_op_qiskit, params[1])
    gate_3 = PauliEvolutionGate(1j * one_body_op_qiskit, params[2])

    qc.append(gate_3, range(4))
    qc.append(gate_2, range(4))
    qc.append(gate_1, range(4))

    return qc


def make_tups_ansatz(n_qubits: int, n_layers: int) -> QuantumCircuit:
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

    for l in range(1, L + 1):
        for p in range(1, B + 1):
            u = make_parameterized_unitary_op(2 * p, 2 * p - 1)

            # if more than one layer prepend parameters with layer number to preserve name uniqueness
            if n_layers != 1:
                u = prepend_params(u, f"l{l}")

            qc.append(u.to_gate(label="U"), range(4 * (p - 1), 4 * p))
        for p in range(1, A + 1):
            u = make_parameterized_unitary_op(2 * p + 1, 2 * p)

            # if more than one layer prepend parameters with layer number to preserve name uniqueness
            if n_layers != 1:
                u = prepend_params(u, f"l{l}")

            qc.append(u.to_gate(label="U"), range(2 + 4 * (p - 1), 2 + 4 * p))

    return qc


def make_ucc_ansatz(
    n_qubits: int, n_electrons: int, n_excitations: int
) -> QuantumCircuit:
    """
    Implements the Unitary Coupled Cluster Ansatz. See https://arxiv.org/pdf/2109.15176

    Args:
        n_qubits: int, the number of qubits of the Ansatz circuit
        n_electrons: int, the number of electrons in the molecule
        n_excitations: int, the number of excitations to truncate coupled cluster at
            must be <= n_electrons
    """

    qc = QuantumCircuit(n_qubits)

    excitations = []
    params = []
    for n in range(1, n_excitations + 1):
        occupied = [*combinations(range(0, n_electrons), n)]
        virtual = [*combinations(range(n_electrons, n_qubits), n)]
        excitations += [
            FermionOperator(
                " ".join(f"{j}^" for j in v) + " " + " ".join(str(j) for j in o)
            )
            for o in occupied
            for v in virtual
        ]
        params += [Parameter(f"n{n}o{o}v{v}") for o in occupied for v in virtual]

    jw_excitations = [jordan_wigner(j) for j in excitations]
    q_excitations = [openfermion_to_qiskit(j, n_qubits) for j in jw_excitations]

    T_terms = [
        1j * (q_ex - q_ex.transpose().conjugate()).simplify() for q_ex in q_excitations
    ]
    trotter_gates = [PauliEvolutionGate(ex, p) for ex, p in zip(T_terms, params)]

    for gate in trotter_gates:
        qc.append(gate, range(n_qubits))

    return qc
