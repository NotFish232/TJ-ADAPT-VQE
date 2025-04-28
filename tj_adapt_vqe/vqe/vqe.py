import numpy as np
from openfermion import MolecularData
from qiskit import transpile  # type: ignore
from qiskit.circuit import QuantumCircuit  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Self

from ..observables.measure import DEFAULT_BACKEND, Measure
from ..observables.observable import HamiltonianObservable, Observable
from ..optimizers.optimizer import Optimizer
from ..utils.ansatz import make_perfect_pair_ansatz, make_tups_ansatz
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
        observables: list[Observable] = [],
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

        self.vqe_it = 1

        self.progress_bar = tqdm()  # type: ignore
        self.progress_bar.set_description_str(self._make_progress_description())

    def _make_ansatz(self: Self) -> QuantumCircuit:
        """
        Generates the original ansatz with the VQE uses, this is overriden in the ADAPTVQE alogirhtm
        """
        ansatz = make_perfect_pair_ansatz(self.n_qubits)
        ansatz.compose(make_tups_ansatz(self.n_qubits, 1), inplace=True)

        return transpile(
            ansatz.decompose(reps=2),
            backend=DEFAULT_BACKEND,
            optimization_level=3,
        )

    def _make_progress_description(self: Self) -> str:
        """
        Returns the description that should be added to the tqdm progress bar, this can be overriden by parents classes for additional functionality,
        i.e. by ADAPTVQE to add more metrics
        """

        last_energy = self.logger.logged_values.get("energy", None)
        last_energy_f = f"{last_energy[-1]:5g}" if last_energy is not None else "NA"

        fci_energy = self.molecule.fci_energy
        fci_energy_f = f"{fci_energy:5g}" if fci_energy is not None else "NA"

        energy_percent_f = (
            f"{100 * last_energy[-1] / fci_energy:5g}%"
            if last_energy is not None and fci_energy is not None
            else "NA"
        )

        last_grad = self.logger.logged_values.get("avg_grad", None)
        last_grad_f = f"{last_grad[-1]:5g}" if last_grad is not None else "NA"

        return f"VQE it: {self.vqe_it} | Energy: {last_energy_f} | FCI: {fci_energy_f} | %: {energy_percent_f} | grad: {last_grad_f}"

    def optimize_parameters(self: Self) -> None:
        """
        Performs a single iteration step of the vqe, stopping when the provided Optimizer's stopping condition has been reached
        """

        while True:
            # perform an iteration of updates
            measure = Measure(
                self.circuit,
                self.param_vals,
                [self.hamiltonian, *self.observables],
                [self.hamiltonian],
                num_shots=self.num_shots,
            )

            h_grad = measure.grads[self.hamiltonian]

            self.param_vals = self.optimizer.update(self.param_vals, h_grad)

            # log important values
            self.logger.add_logged_value("energy", measure.evs[self.hamiltonian])

            for obv in self.observables:
                self.logger.add_logged_value(obv.name, measure.evs[obv])

            self.logger.add_logged_value("params", self.param_vals.tolist(), file=True)
            self.logger.add_logged_value("avg_grad", np.mean(np.abs(h_grad)))
            self.logger.add_logged_value("max_grad", np.max(np.abs(h_grad)))

            self.progress_bar.update()
            self.progress_bar.set_description_str(self._make_progress_description())

            if self.optimizer.is_converged(h_grad):
                break

            self.vqe_it += 1
