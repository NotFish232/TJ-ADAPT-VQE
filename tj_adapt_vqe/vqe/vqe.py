from math import log10

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from openfermion import MolecularData
from qiskit import QuantumCircuit, qasm3, transpile  # type: ignore
from qiskit_aer import AerSimulator  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Any, Self

from ..observables.measure import (
    EXACT_BACKEND,
    Measure,
    make_ev_function,
    make_grad_function,
)
from ..observables.observable import HamiltonianObservable, Observable
from ..optimizers.optimizer import (
    FunctionalOptimizer,
    GradientOptimizer,
    HybridOptimizer,
    NonGradientOptimizer,
    Optimizer,
    OptimizerType,
)
from ..utils.ansatz import Ansatz
from ..utils.logger import Logger


class VQE:
    """
    Implements the Variational Quantum Eigensolver (VQE) algorithm.
    """

    def __init__(
        self: Self,
        molecule: MolecularData,
        optimizer: Optimizer,
        starting_ansatz: list[Ansatz] = [],
        observables: list[Observable] = [],
        qiskit_backend: AerSimulator = EXACT_BACKEND,
    ) -> None:
        """
        Constructs an instance of `VQE`. Sets object properties and intializes the starting ansatz
        through a call to `self._make_ansatz(...)` and transpiles the circuit to work on the qiskit backend specified.
        Also initializes the logger class to log both config data and metrics from training.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule to run the VQE algorithm on.
            optimizer (Optimizer): The optimizer used to update parameter values at each step.
            starting_ansatz (list[Ansatz]): A list of the starting ansatz that should be used.
            observables (list[Observable], optional): The observables to monitor the values of. Defaults to [].
            qiskit_backend: AerSimulator. Backend to run measures on. Defaults to EXACT_BACKEND.
        """

        self.molecule = molecule
        self.hamiltonian = HamiltonianObservable(molecule)
        self.n_qubits = self.molecule.n_qubits

        self.optimizer = optimizer
        self.observables = observables

        self.qiskit_backend = qiskit_backend
        self.starting_ansatz = starting_ansatz
        self.circuit = self._make_ansatz()
        self.transpiled_circuit = self._transpile_circuit(self.circuit)

        n_params = len(self.circuit.parameters)
        self.param_vals = (2 * np.random.rand(n_params) - 1) / np.sqrt(n_params)

        self.logger = Logger(self._run_information())
        self.logger.start()

        self.logger.add_config_option("molecule", self.molecule.name)
        self.logger.add_config_option("optimizer", self.optimizer.to_config())
        self.logger.add_config_option("starting_ansatz", self.starting_ansatz)
        self.logger.add_config_option(
            "qiskit_backend", self.qiskit_backend.options.__dict__
        )

        self.vqe_it = 0

        self.progress_bar: tqdm = None  # type: ignore

    def _run_information(self: Self) -> str:
        """
        Returns the run information used for the run name in logger. Should omit the class name,
        can be override by parent classes.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            str: A descriptive string of the current configuration.
        """

        return f"{self.optimizer.name} {self.molecule.name}"

    def _transpile_circuit(self: Self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Transpiles a circuit based on the backend in `self.qiskit_backend` and with maximized optimization.

        Args:
            self (Self): A reference to the current class instance.
            qc (QuantumCircuit): The quantum circuit to optimize.

        Returns:
            QuantumCircuit: The transpiled quantum circuit.
        """

        return transpile(qc, backend=self.qiskit_backend, optimization_level=3)

    def _make_ansatz(self: Self) -> QuantumCircuit:
        """
        Makes the ansatz for the VQE algorithm based on `self.starting_ansatz`.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            QuantumCircuit: The ansatz to use for the VQE algorithm.
        """

        qc = QuantumCircuit(self.n_qubits)

        for ansatz in self.starting_ansatz:
            qc.compose(ansatz.construct(self.molecule), inplace=True)

        return qc

    def _make_progress_description(self: Self) -> str:
        """
        Returns a string that is used as the description of the progress bar during training.
        Can be overriden by subclasses to include more information concatenated to a call to
        `super()._make_progress_description(...)`.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            str: The progress bar description.
        """

        last_energy = self.logger.logged_values.get("energy", None)
        last_energy_f = f"{last_energy[-1]:5g}" if last_energy is not None else "NA"

        fci_energy = self.molecule.fci_energy
        fci_energy_f = f"{fci_energy:5g}" if fci_energy is not None else "NA"

        energy_percent_f = (
            f"{abs(fci_energy - last_energy[-1]):e}"
            if last_energy is not None and fci_energy is not None
            else "NA"
        )

        last_grad = self.logger.logged_values.get("grad", None)
        last_grad_f = f"{last_grad[-1]:5g}" if last_grad is not None else "NA"

        return f"VQE it: {self.vqe_it} | Energy: {last_energy_f} | FCI: {fci_energy_f} | %: {energy_percent_f} | grad: {last_grad_f}"

    def _perform_step(self: Self) -> bool:
        """
        Performs a single step of optimization, calculating the values required for the selected optimizer type.
        Returns whether or not the optimization has converged. Can be called if `self.optimizer.type` is anything
        besides `OptimizerType.Functional`.

        Args:
            self (Self): A reference to the current class instance.

        Raises:
            NotImplementedError: Each possible subclass of `Optimizer` was not exhaustively checked and returned out of.

        Returns:
            bool: Whether the VQE algorithm has converged.

        """

        # observables to calculate evs and grads of
        ev_observables: list[Observable] = [self.hamiltonian]
        grad_observables: list[Observable] = []
        if self.optimizer.type in (OptimizerType.Gradient, OptimizerType.Hybrid):
            grad_observables.append(self.hamiltonian)

        m = Measure(
            self.transpiled_circuit,
            self.param_vals,
            ev_observables,
            grad_observables,
            qiskit_backend=self.qiskit_backend,
        )

        # log hamiltonian gradients
        if (h_grad := m.grads.get(self.hamiltonian)) is not None:
            self.logger.add_logged_value("avg_grad", np.mean(np.abs(h_grad)))
            self.logger.add_logged_value("max_grad", np.max(np.abs(h_grad)))

        if isinstance(self.optimizer, GradientOptimizer):
            grad = m.grads[self.hamiltonian]

            self.param_vals = self.optimizer.update(self.param_vals, grad)

            return self.optimizer.is_converged(grad)

        elif isinstance(self.optimizer, NonGradientOptimizer):
            f = make_ev_function(
                self.transpiled_circuit, self.hamiltonian, self.qiskit_backend
            )

            self.param_vals = self.optimizer.update(self.param_vals, f)

            return self.optimizer.is_converged()

        elif isinstance(self.optimizer, HybridOptimizer):
            grad = m.grads[self.hamiltonian]
            f = make_ev_function(
                self.transpiled_circuit, self.hamiltonian, self.qiskit_backend
            )

            self.param_vals = self.optimizer.update(self.param_vals, grad, f)

            return self.optimizer.is_converged(grad)

        raise NotImplementedError()

    def _vqe_iteration_hook(
        self: Self, param_vals: np.ndarray, *args: tuple[Any]
    ) -> None:
        """
        A callback that should be called at each step of the optimization process.
        Necessary for our VQE interface to be compatible with both in place and functional
        optimizers.

        Args:
            self (Self): A reference to the current class instance.
            param_vals (np.ndarray): The new parameter values.
            *args (tuple[Any]): Unused excess arguments to make function signature compatible.
        """

        self.param_vals = param_vals

        m = Measure(
            self.transpiled_circuit,
            self.param_vals,
            [self.hamiltonian, *self.observables],
            qiskit_backend=self.qiskit_backend,
        )

        # log molecular energies
        self.logger.add_logged_value("energy", m.evs[self.hamiltonian])
        if self.molecule.fci_energy is not None:
            energy_p = abs(m.evs[self.hamiltonian] - self.molecule.fci_energy)
            energy_p_log = log10(energy_p)
            self.logger.add_logged_value("energy_percent", energy_p)
            self.logger.add_logged_value("energy_percent_log", energy_p_log)

        # log observable quantities
        for obv in self.observables:
            self.logger.add_logged_value(obv.name, m.evs[obv])

        self.progress_bar.update()
        self.progress_bar.set_description_str(self._make_progress_description())  # type: ignore

        self.vqe_it += 1

    def run(self: Self, show_pbar: bool = True) -> None:
        """
        Runs an iteration of the VQE algorithm, optimizing parameters to minimize the expectation
        value of the hamiltonian until a stopping condition, which is determined by the optimizer,
        is met.

        Args:
            self (Self): A reference to the current class instance.
            show_pbar (bool): Whether to show progress bar. Defaults to True.
        """

        # creates progress bar if not created
        # assert ownership of it
        if self.progress_bar is None:
            self.progress_bar = tqdm(disable=not show_pbar)  # type: ignore
            created_pbar = True
        else:
            created_pbar = False

        self.logger.add_logged_value(
            "ansatz_qasm", qasm3.dumps(self.transpiled_circuit), file=True
        )
        fig = self.transpiled_circuit.draw("mpl")
        self.logger.add_logged_value("ansatz_img", fig, file=True)
        # close figure manually
        plt.close(fig)

        # call hook manually first time
        self._vqe_iteration_hook(self.param_vals)

        if isinstance(self.optimizer, FunctionalOptimizer):
            # perform entire optimization in one step
            self.optimizer.update(
                self.param_vals,
                make_ev_function(
                    self.transpiled_circuit, self.hamiltonian, self.qiskit_backend
                ),
                make_grad_function(
                    self.transpiled_circuit, self.hamiltonian, self.qiskit_backend
                ),
                self._vqe_iteration_hook,
            )
        else:
            while not self._perform_step():
                self._vqe_iteration_hook(self.param_vals)

        self.logger.add_logged_value("params", self.param_vals.tolist(), file=True)

        if created_pbar:
            self.logger.end()
            self.progress_bar.close()
