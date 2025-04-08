from .optimizers import SGD
from .pools import FSD
from .utils import AvailableMolecules
from .utils.molecules import make_molecule
from .vqe import VQE, ADAPTVQE


def main() -> None:
    # h2 = make_molecule(AvailableMolecules.H2, r=1.5)
    # optimizer = SGD()
    # vqe = VQE(h2, optimizer)
    # vqe.run()

    h2 = make_molecule(AvailableMolecules.H2, r=1.5)
    optimizer = SGD()
    adapt = ADAPTVQE(h2, FSD(h2, 2), optimizer)
    adapt.run()


if __name__ == "__main__":
    main()
