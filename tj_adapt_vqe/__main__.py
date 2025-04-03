from tj_adapt_vqe.optimizers import SGD
from tj_adapt_vqe.utils import AvailableMolecules
from tj_adapt_vqe.utils.molecules import make_molecule
from tj_adapt_vqe.vqe import VQE


def main() -> None:
    molecule = make_molecule(AvailableMolecules.H2, r=1.5)

    optimizer = SGD()

    vqe = VQE(molecule, optimizer)

    vqe.optimize_parameters()
    
 


if __name__ == "__main__":
    main()
