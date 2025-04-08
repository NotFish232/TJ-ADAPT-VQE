from .observables import (
    NumberObservable,
    Observable,
    SpinSquaredObservable,
    SpinZObservable,
)
from .optimizers import SGD
from .pools import FSD
from .utils import Molecule, make_molecule
from .vqe import ADAPTVQE


def main() -> None:
    h2 = make_molecule(Molecule.H2, r=1.5)

    optimizer = SGD()

    observables: list[Observable] = [
        NumberObservable(h2.n_qubits),
        SpinZObservable(h2.n_qubits),
        SpinSquaredObservable(h2.n_qubits),
    ]

    adapt = ADAPTVQE(h2, FSD(h2, 2), optimizer, observables)
    adapt.run()


if __name__ == "__main__":
    main()
