from openfermion import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore
from typing_extensions import Protocol, Self, Type, override

from .serializable import Serializable


class MoleculeConstructor(Protocol):
    def __call__(self: Self, r: float, run_fci: bool = True) -> "Molecule": ...


class Molecule(Serializable):
    """
    A wrapper around the openfermion `MolecularData` class that allows serialization.
    """

    H2: MoleculeConstructor
    H2_631G: MoleculeConstructor
    H4: MoleculeConstructor
    H4_631G: MoleculeConstructor
    H5: MoleculeConstructor
    H6: MoleculeConstructor
    LiH: MoleculeConstructor
    BeH2: MoleculeConstructor

    _all: list[MoleculeConstructor] = []

    def __init__(
        self: Self,
        name: str,
        geometry: list[tuple[str, tuple[int, int, int]]],
        basis: str,
        multiplicity: int,
        charge: int,
        r: float,
        run_fci: bool = True,
    ) -> None:
        """
        Args:
            self (Self): A reference to the current class instance.
            name (str): The name of the molecule.
            geometry (list[tuple[str, tuple[int, int, int]]]): The geometry, scaled by r.
            basis (str): The basis functions for the molecule.
            multiplicity (int): Molecule's multiplicity.
            charge (int): Molecule's charge.
            r (float): A radius which the geometry is scaled by.
            run_fci (bool): Whether to run_fci on the molecule
        """

        self.name = name
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.r = r
        self.run_fci = run_fci

        scaled_geometry = [(a, (r * x, r * y, r * z)) for a, (x, y, z) in geometry]

        self.data = MolecularData(scaled_geometry, basis, multiplicity, charge, name)

        if run_fci:
            self.data = run_pyscf(self.data, run_fci=True)

    @staticmethod
    @override
    def _type() -> str:
        """
        Returns the type of the class. Used in `Serializable`.
        """

        return "molecule"

    @staticmethod
    @override
    def _name() -> str:
        """
        Returns the name of the class. Used in `Serializable`.
        """

        return ""

    @property
    @override
    def _config_params(self: Self) -> list[str]:
        """
        Returns a list of config parameters.
        """

        return ["name", "geometry", "basis", "multiplicity", "charge", "r", "run_fci"]

    @classmethod
    @override
    def all(cls: Type["Molecule"]) -> list[Type["Molecule"]]:
        raise NotImplementedError(
            "For non-abstract classes, use all_constructors instead()."
        )

    @classmethod
    @override
    def all_constructors(cls: Type["Molecule"]) -> list[MoleculeConstructor]:
        """
        Little bit suspect
        """

        return Molecule._all


def _molecule_constructor_wrapper(
    name: str,
    geometry: list[tuple[str, tuple[int, int, int]]],
    basis: str,
    multiplicity: int,
    charge: int,
) -> MoleculeConstructor:
    """
    Wraps the constructor of the Molecule class.

    Returns:
        MoleculeConstructor: A new constructor that only takes a radius and run_fci.
    """

    def _callable(r: float, run_fci: bool = True) -> Molecule:
        return Molecule(name, geometry, basis, multiplicity, charge, r, run_fci)

    Molecule._all.append(_callable)

    return _callable


Molecule.H2 = _molecule_constructor_wrapper(
    name="H2",
    geometry=[("H", (0, 0, 0)), ("H", (0, 0, 1))],
    basis="sto-3g",
    multiplicity=1,
    charge=0,
)
Molecule.H2_631G = _molecule_constructor_wrapper(
    name="H2",
    geometry=[("H", (0, 0, 0)), ("H", (0, 0, 1))],
    basis="6-31g",
    multiplicity=1,
    charge=0,
)
Molecule.H4 = _molecule_constructor_wrapper(
    name="H4",
    geometry=[
        ("H", (0, 0, 0)),
        ("H", (0, 0, 1)),
        ("H", (0, 0, 2)),
        ("H", (0, 0, 3)),
    ],
    basis="sto-3g",
    multiplicity=1,
    charge=0,
)
Molecule.H4_631G = _molecule_constructor_wrapper(
    name="H4",
    geometry=[
        ("H", (0, 0, 0)),
        ("H", (0, 0, 1)),
        ("H", (0, 0, 2)),
        ("H", (0, 0, 3)),
    ],
    basis="6-31g",
    multiplicity=1,
    charge=0,
)
Molecule.H5 = _molecule_constructor_wrapper(
    name="H5",
    geometry=[
        ("H", (0, 0, 0)),
        ("H", (0, 0, 1)),
        ("H", (0, 0, 2)),
        ("H", (0, 0, 3)),
        ("H", (0, 0, 4)),
    ],
    basis="sto-3g",
    multiplicity=2,
    charge=0,
)
Molecule.H6 = _molecule_constructor_wrapper(
    name="H6",
    geometry=[
        ("H", (0, 0, 0)),
        ("H", (0, 0, 1)),
        ("H", (0, 0, 2)),
        ("H", (0, 0, 3)),
        ("H", (0, 0, 4)),
        ("H", (0, 0, 5)),
    ],
    basis="sto-3g",
    multiplicity=1,
    charge=0,
)
Molecule.LiH = _molecule_constructor_wrapper(
    name="LiH",
    geometry=[("Li", (0, 0, 0)), ("H", (0, 0, 1))],
    basis="sto-3g",
    multiplicity=1,
    charge=0,
)
Molecule.BeH2 = _molecule_constructor_wrapper(
    name="BeH2",
    geometry=[("Be", (0, 0, 0)), ("H", (0, 0, 1)), ("H", (0, 0, -1))],
    basis="sto-3g",
    multiplicity=1,
    charge=0,
)
