from .optimizers import SGD
from .pools import FSD
from .utils import Molecule, make_molecule
from .vqe import ADAPTVQE


def main() -> None:
    h2 = make_molecule(Molecule.H2, r=1.5)

    optimizer = SGD()
    adapt = ADAPTVQE(h2, FSD(h2, 2), optimizer)
    adapt.run()


if __name__ == "__main__":
    main()
