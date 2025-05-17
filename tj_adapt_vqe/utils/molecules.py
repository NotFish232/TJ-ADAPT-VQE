from collections import namedtuple
from enum import Enum

from openfermion import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore

MoleculeInfo = namedtuple(
    "MoleculeInfo", ["geometry", "basis", "multiplicity", "charge"]
)


class Molecule(Enum):
    H2 = MoleculeInfo(
        geometry=lambda r: [["H", [0, 0, 0]], ["H", [0, 0, r]]],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
    )
    H4 = MoleculeInfo(
        geometry=lambda r: [
            ("H", (0, 0, 0)),
            ("H", (0, 0, r)),
            ("H", (0, 0, 2 * r)),
            ("H", (0, 0, 3 * r)),
        ],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
    )
    H5 = MoleculeInfo(
        geometry=lambda r: [
            ("H", (0, 0, 0)),
            ("H", (0, 0, r)),
            ("H", (0, 0, 2 * r)),
            ("H", (0, 0, 3 * r)),
            ("H", (0, 0, 4 * r)),
        ],
        basis="sto-3g",
        multiplicity=2,
        charge=0,
    )
    H6 = MoleculeInfo(
        geometry=lambda r: [
            ("H", (0, 0, 0)),
            ("H", (0, 0, r)),
            ("H", (0, 0, 2 * r)),
            ("H", (0, 0, 3 * r)),
            ("H", (0, 0, 4 * r)),
            ("H", (0, 0, 5 * r)),
        ],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
    )
    LiH = MoleculeInfo(
        geometry=lambda r: [["Li", [0, 0, 0]], ["H", [0, 0, r]]],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
    )
    BeH2 = MoleculeInfo(
        geometry=lambda r: [["Be", [0, 0, 0]], ["H", [0, 0, r]], ["H", [0, 0, -r]]],
        basis="sto-3g",
        multiplicity=1,
        charge=0,
    )


def make_molecule(molecule: Molecule, r: float, run_fci: bool = True) -> MolecularData:
    # geometry is a lambda to support arbitrary r
    molecule_dict = molecule.value._asdict()
    molecule_dict["geometry"] = molecule_dict["geometry"](r)

    openfermion_molecule = MolecularData(**molecule_dict, description=molecule.name)

    if run_fci:
        openfermion_molecule = run_pyscf(
            openfermion_molecule, run_fci=True, run_ccsd=False
        )

    return openfermion_molecule
