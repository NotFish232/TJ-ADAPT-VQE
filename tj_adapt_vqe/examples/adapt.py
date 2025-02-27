"""
A file that shows an example of how to use the ADAPTVQE class to run a quantum simulation using the library.
"""


def main():
    molecule = MolecularData()
    vqe = VQE(molecule)
    vqe.run()
    train(vqe)


if __name__ == "__main__":
    main()
