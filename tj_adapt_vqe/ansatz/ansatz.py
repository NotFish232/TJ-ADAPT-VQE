from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit  # type: ignore
from typing_extensions import Self, override

from ..utils.molecules import Molecule
from ..utils.serializable import Serializable
from .functional import (
    make_hartree_fock_ansatz,
    make_perfect_pair_ansatz,
    make_qiskit_uccsd,
    make_tups_ansatz,
    make_ucc_ansatz,
)


class Ansatz(Serializable, ABC):
    """
    Inherits from `Serializable` and `abc.ABC`. Base class for all other Ansatzes. Provides
    a class interface that uses the functional ones, which allows it to be saved and loaded.
    """

    @staticmethod
    @override
    def _type() -> str:
        """
        Returns the type of this class. Used in `Serializable`.
        """

        return "ansatz"

    @abstractmethod
    def construct(self: Self, molecule: Molecule) -> QuantumCircuit:
        """
        Constructs an instance of the associated ansastz given a molecule.

        Args:
            self (Self): A reference to the current class instance.
            molecule (Molecule): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The quantum circuit associated with the ansatz and the molecule.
        """

        pass


class HartreeFockAnsatz(Ansatz):
    """
    Inherits from `Ansatz`. The HartreeFock ansatz.
    """

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "hartree_fock_ansatz"

    @override
    def construct(self: Self, molecule: Molecule) -> QuantumCircuit:
        """
        Generates the hartree fock ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (Molecule): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The hartree fock quantum circuit.
        """

        return make_hartree_fock_ansatz(
            molecule.data.n_qubits, molecule.data.n_electrons
        )


class PerfectPairAnsatz(Ansatz):
    """
    Inherits from `Ansatz`. The PerfectPair ansatz.
    """

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "perfect_pair_ansatz"

    @override
    def construct(self: Self, molecule: Molecule) -> QuantumCircuit:
        """
        Generates the perfect pair ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (Molecule): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The perfect pairing quantum circuit.
        """

        return make_perfect_pair_ansatz(molecule.data.n_qubits)


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

        self.n_layers = n_layers

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "tups_ansatz"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["n_layers"]

    @override
    def construct(self: Self, molecule: Molecule) -> QuantumCircuit:
        """
        Generates the tups ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (Molecule): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The TUPS quantum circuit.
        """

        return make_tups_ansatz(molecule.data.n_qubits, self.n_layers)


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

        self.n_excitations = n_excitations

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "ucc_ansatz"

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns the config attributes of this class. Used in `Serializable`.
        """

        return ["n_excitations"]

    @override
    def construct(self: Self, molecule: Molecule) -> QuantumCircuit:
        """
        Generates the UCC ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (Molecule): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The UCC quantum circuit.
        """

        return make_ucc_ansatz(
            molecule.data.n_qubits, molecule.data.n_electrons, self.n_excitations
        )


class QiskitUCCSDAnsatz(Ansatz):
    """
    Inherits from `Ansatz`. The UCC ansatz using Qiskit's implementation.
    """

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of this class. Used in `Serializable`.
        """

        return "qiskit_uccsd_ansatz"

    @override
    def construct(self: Self, molecule: Molecule) -> QuantumCircuit:
        """
        Generates the UCC ansatz through the functional interface.

        Args:
            self (Self): A reference to the current class instance.
            molecule (Molecule): The molecule that the ansatz should be constructed from.

        Returns:
            QuantumCircuit: The UCC quantum circuit.
        """

        return make_qiskit_uccsd(molecule.data.n_qubits, molecule.data.n_electrons)
