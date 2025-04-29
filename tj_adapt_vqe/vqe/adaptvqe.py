import numpy as np
from openfermion import MolecularData
from qiskit.circuit import Parameter, QuantumCircuit  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Self, override

from ..observables import Observable, SparsePauliObservable
from ..observables.measure import Measure
from ..optimizers import Optimizer
from ..pools import Pool
from ..utils.ansatz import make_hartree_fock_ansatz
from .vqe import VQE


class ADAPTVQE(VQE):
    """
    Class implementing the ADAPT-VQE algorithm
    """

    def __init__(
        self: Self,
        molecule: MolecularData,
        pool: Pool,
        optimizer: Optimizer,
        observables: list[Observable] = [],
        num_shots: int = 1024,
        op_gradient_convergence_threshold: float = 0.01,
    ) -> None:
        """
        Initializes the ADAPTVQE object
        Args:
            molecule: MolecularData, the molecular data that is used for the adapt vqe
            pool: Pool, the pool that the ADAPTVQE uses to form the Ansatz
            op_gradient_convergence_threshold: float = 1e-2, the convergence criterion for the adapt vqe algorithm
            Other arguments are passed directly to the VQE constructor
        """

        self.adapt_vqe_it = 1

        super().__init__(molecule, optimizer, observables, num_shots)

        self.pool = pool

        self.commutators = self._calculate_commutators()

        self.op_gradient_convergence_threshold = op_gradient_convergence_threshold

        self.logger.add_config_option("pool", self.pool.to_config())

    @override
    def _make_ansatz(self: Self) -> QuantumCircuit:
        """
        Overides the VQE ansatz by starting it unparameterized
        """
        ansatz = make_hartree_fock_ansatz(self.n_qubits, self.molecule.n_electrons)

        return self._transpile_circuit(ansatz)

    @override
    def _make_progress_description(self: Self) -> str:
        """
        Overrides the VQE progress_description including all of its params along with ADAPTVQE specific details
        """

        vqe_progress_descrption = super()._make_progress_description()

        n_params_f = f"{len(self.circuit.parameters)}"

        return f"ADAPT-VQE it: {self.adapt_vqe_it} | {vqe_progress_descrption} | N-Params: {n_params_f}"

    def _calculate_commutators(self: Self) -> list[Observable]:
        """
        Calculates the commutator between the Hamiltonian and every Observable
        within the provided Pool
        """

        H = self.hamiltonian.operator

        return [
            SparsePauliObservable(
                (1j * (H @ A - A @ H).simplify()).simplify(),
                f"commutator_{i}",
                self.n_qubits,
            )
            for i, A in enumerate(self.pool.operators)
        ]

    def _find_best_operator(self: Self) -> tuple[float, int]:
        """
        Returns the gradient and idx of the best operator  within the pool that maximizes the commutator with the hamiltonian

        """

        m = Measure(
            self.circuit,
            self.param_vals,
            self.commutators,
            num_shots=self.num_shots,
        )

        grads = np.abs([m.evs[c] for c in self.commutators])

        idx = np.argmax(grads).item()

        return grads[idx], idx

    def run(self: Self) -> None:
        """
        Runs the ADAPT-VQE Algorithm
        """

        # creates progress bar if not created
        # assert ownership of it
        if self.progress_bar is None:
            self.progress_bar = tqdm()  # type: ignore
            self.progress_bar.set_description_str(self._make_progress_description())
            created_pbar = True
        else:
            created_pbar = False

        while True:
            max_grad, max_idx = self._find_best_operator()

            if max_grad < self.op_gradient_convergence_threshold:
                break

            new_op = self.pool.operators[max_idx]
            new_param = Parameter(f"n{self.adapt_vqe_it}{self.pool.labels[max_idx]}")

            self.param_vals = np.append(self.param_vals, np.random.rand(1) - 0.5)
            self.circuit.compose(PauliEvolutionGate(new_op, new_param), inplace=True)
            self.circuit = self._transpile_circuit(self.circuit)

            self.logger.add_logged_value("new_operator", max_idx)
            self.logger.add_logged_value("new_operator_grad", max_grad)
            self.logger.add_logged_value("n_params", len(self.param_vals), t=self.vqe_it)

            super().run()
            self.optimizer.reset()

            self.adapt_vqe_it += 1

        if created_pbar:
            self.progress_bar.close()
