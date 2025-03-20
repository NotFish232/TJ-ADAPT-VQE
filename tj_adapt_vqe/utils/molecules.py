from enum import Enum

from openfermion import MolecularData
from openfermionpyscf import run_pyscf  # type: ignore


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
