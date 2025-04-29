import numpy as np
from openfermion import MolecularData
from qiskit.circuit import QuantumCircuit  # type: ignore
from typing_extensions import Self

from ..observables.measure import Measure
from ..observables.observable import HamiltonianObservable, Observable
from ..optimizers.optimizer import Optimizer
from ..utils.ansatz import make_perfect_pair_ansatz, make_tups_ansatz, make_ucc_ansatz
from ..utils.logger import Logger


class VQE:
    """
    Class implementing the variational quantum eigensolver (VQE) algorithm

    Args:
        molecule: Moleculardata, molecule to find ground state of
        optimizer: Optimizer, optimizer that the Measure class is passed into
        observables: list[Oobservable], what observables should be calculated each iteration
        num_shots: int, num shots to run each simulation with

    """

    def __init__(
        self: Self,
        molecule: MolecularData,
        optimizer: Optimizer,
        observables: list[Observable],
        num_shots: int = 1024,
    ) -> None:

        self.molecule = molecule
        self.hamiltonian = HamiltonianObservable(molecule)
        self.n_qubits = self.molecule.n_qubits

        self.optimizer = optimizer

        self.observables = observables

        self.num_shots = num_shots

        self.circuit = self._make_ansatz()

        self.param_vals = 2 * np.random.rand(len(self.circuit.parameters)) - 1

        self.logger = Logger()

        self.logger.add_config_option("optimizer", self.optimizer.to_config())
        self.logger.add_config_option("molecule", self.molecule.name)

    def _make_ansatz(self: Self) -> QuantumCircuit:
        ansatz = make_perfect_pair_ansatz(self.n_qubits).compose(
            # make_tups_ansatz(self.n_qubits, 1)
            make_ucc_ansatz(self.n_qubits, self.molecule.n_electrons, 2)
        )

        return ansatz.decompose(reps=2)

    def optimize_parameters(self: Self) -> None:
        """
        Performs a single iteration step of the vqe, stopping when the provided Optimizer's stopping condition has been reached
        """

        iteration = 1

        while True:
            measure = Measure(
                self.circuit,
                self.param_vals,
                [self.hamiltonian, *self.observables],
                [self.hamiltonian],
                num_shots=self.num_shots,
            )

            self.logger.add_logged_value("energy", measure.evs[self.hamiltonian])

            for obv in self.observables:
                self.logger.add_logged_value(obv.name, measure.evs[obv])

            h_grad = measure.grads[self.hamiltonian]

            self.logger.add_logged_value("params", self.param_vals.tolist(), file=True)
            self.logger.add_logged_value("avg_grad", np.mean(np.abs(h_grad)))
            self.logger.add_logged_value("max_grad", np.max(np.abs(h_grad)))

            iteration += 1

            self.param_vals = self.optimizer.update(self.param_vals, h_grad)

            if self.optimizer.is_converged(h_grad):
                break
