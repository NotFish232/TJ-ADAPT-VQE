from tj_adapt_vqe.optimizers import SGDOptimizer
from tj_adapt_vqe.utils import AvailableMolecules, Measure, make_molecule
from tj_adapt_vqe.vqe import VQE


def main() -> None:
    h2 = make_molecule(AvailableMolecules.H2, r=1.5)

    optimizer = SGDOptimizer()

    vqe = VQE(h2, optimizer)

    LR = 0.5

    for i in range(1000):
        measure = Measure(vqe.circuit, vqe.param_values, vqe.molecular_hamiltonian_qiskit)

        print(f"ev = {measure.expectation_value}, p = {vqe.param_values}, g = {measure.gradients}")
        print()

        vqe.param_values -= LR * measure.gradients


if __name__ == "__main__":
    main()
