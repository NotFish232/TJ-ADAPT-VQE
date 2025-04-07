from .optimizers import SGD
from .utils import Molecule, make_molecule
from .vqe import VQE


def main() -> None:
    molecule = make_molecule(Molecule.H2, r=1.5)

    optimizer = SGD()

    vqe = VQE(molecule, optimizer)

    vqe.optimize_parameters()


if __name__ == "__main__":
    main()
