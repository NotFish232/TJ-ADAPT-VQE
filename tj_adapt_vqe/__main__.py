from tj_adapt_vqe.optimizers import SGDOptimizer
from tj_adapt_vqe.utils import AvailableMolecules, make_molecule
from tj_adapt_vqe.vqe import VQE


def main() -> None:
    h2 = make_molecule(AvailableMolecules.H2, r=1.5)

    optimizer = SGDOptimizer()

    vqe = VQE(h2, optimizer)

    # TODO FIXME: how do I find the expectation value of an observable on a qiskit circuit
    # without doing it clasically and just extracting the actual quantum state



if __name__ == "__main__":
    main()
