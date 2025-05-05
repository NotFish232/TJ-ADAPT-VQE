from enum import Enum

import numpy as np
from openfermion import MolecularData
from qiskit.circuit import Parameter, QuantumCircuit, Gate  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Self, override

from ..observables import Observable, SparsePauliObservable
from ..observables.measure import Measure
from ..optimizers import Optimizer
from ..pools import Pool
from ..utils.ansatz import make_perfect_pair_ansatz
from .vqe import VQE


class ADAPTConvergenceCriteria(Enum):
    """
    Criteria used to evaluate when to stop the ADAPT-VQE algorithm
    Possible criteria includes Gradient, whether the max gradient of any pool operators is below a threshold
    Or whether not enough improvement has been made in successive iterations.
    """

    Gradient = 0
    LackOfImprovement = 1


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
        adapt_conv_criteria: ADAPTConvergenceCriteria = ADAPTConvergenceCriteria.Gradient,
        conv_threshold: float = 0.01,
    ) -> None:
        """
        Initializes the ADAPTVQE object

        Args:
            molecule: MolecularData, the molecular data that is used for the adapt vqe,
            pool: Pool, the pool that the ADAPTVQE uses to form the Ansatz,
            adapt_conv_criteria: ADAPTConvergenceCriteria = ADAPTConvergenceCriteria.Gradient, the criteria for
            when to terminate adaptive iterations, see the actual Enum impl for information about each criteria,
            conv_threshold: float = 0.01, the threshold wrt to the adapt_conv_criteria to stop adaptive iterations,
            Other arguments are passed directly to the VQE constructor
        """

        self.adapt_vqe_it = 1

        super().__init__(molecule, optimizer, observables, num_shots)

        self.pool = pool

        self.commutators, self.commutator_op_counts = self._calculate_commutators()

        self.adapt_conv_criteria = adapt_conv_criteria
        self.conv_threshold = conv_threshold

        self.logger.add_config_option("pool", self.pool.to_config())

    @override
    def _make_ansatz(self: Self) -> QuantumCircuit:
        """
        Overides the VQE ansatz by starting it unparameterized
        """
        ansatz = make_perfect_pair_ansatz(self.n_qubits)

        return ansatz

    @override
    def _make_progress_description(self: Self) -> str:
        """
        Overrides the VQE progress_description including all of its params along with ADAPTVQE specific details
        """

        vqe_progress_descrption = super()._make_progress_description()

        n_params_f = f"{len(self.circuit.parameters)}"

        return f"ADAPT-VQE it: {self.adapt_vqe_it} | {vqe_progress_descrption} | N-Params: {n_params_f}"

    def _calculate_commutators(self: Self) -> tuple[list[Observable], list[int]]:
        """
        Calculates the commutator between the Hamiltonian and every Observable
        within the provided Pool. Also returnns the "density" for each operator, which is how many actual operators
        each operator within the pool represents. For instance, pool.get_op() could return an array of length 3
        with 3 actual operators that get fused together in poolget_exp_op() but gradients should be calculated seperately
        """

        # if pool only has one operator then clearly commutators aren't necessary
        # used for pools like the TUPS full pool
        if len(self.pool) == 1:
            return [], []

        H = self.hamiltonian.operator

        commutators: list[Observable] = []
        commutator_op_counts = []

        for i in range(len(self.pool)):
            ops = self.pool.get_op(i)

            if isinstance(ops, LinearOp):
                ops = [ops]

            for op in ops:
                commutator = SparsePauliObservable(
                    (H @ op - op @ H).simplify(),
                    f"commutator_{i}",
                )

                commutators.append(commutator)

            commutator_op_counts.append(len(ops))

        return commutators, commutator_op_counts

    def _find_best_operator(self: Self) -> tuple[float, int]:
        """
        Returns the gradient and idx of the best operator  within the pool that maximizes the commutator with the hamiltonian

        """

        # again pool only has one operator, so just return idx 0
        if len(self.pool) == 1:
            return 1, 0

        m = Measure(
            self.transpiled_circuit,
            self.param_vals,
            self.commutators,
            num_shots=self.num_shots,
        )

        grads = []

        i = 0
        for n in self.commutator_op_counts:
            com_grads = []
            for com_idx in range(i, i + n):
                com_grads.append(abs(m.evs[self.commutators[com_idx]]))

            grads.append(np.sum(com_grads).item())

            i += n

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

            # convergence checks, seems hard to seperate this into its own function
            if self.adapt_conv_criteria == ADAPTConvergenceCriteria.Gradient:
                if max_grad < self.conv_threshold:
                    break
            elif self.adapt_conv_criteria == ADAPTConvergenceCriteria.LackOfImprovement:
                # distance betweeen the last two energies from adapt
                if (
                    self.adapt_vqe_it > 2
                    and abs(
                        self.logger.logged_values["energy_adapt"][-2]
                        - self.logger.logged_values["energy_adapt"][-1]
                    )
                    < self.conv_threshold
                ):
                    break

            new_op = self.pool.get_exp_op(max_idx)
            op_label = self.pool.get_label(max_idx)

            if isinstance(new_op, Gate):
                new_op = QuantumCircuit(new_op.num_qubits).compose(new_op)

            new_op.assign_parameters(
                {
                    p: Parameter(f"n{self.adapt_vqe_it}{op_label}{p.name}")
                    for p in new_op.parameters
                },
                inplace=True,
            )

            self.param_vals = np.append(
                self.param_vals, np.zeros(len(new_op.parameters))
            )

            self.circuit.compose(new_op, inplace=True)
            self.transpiled_circuit = self._transpile_circuit(self.circuit)

            self.logger.add_logged_value("new_operator", max_idx)
            self.logger.add_logged_value("new_operator_grad", max_grad)
            self.logger.add_logged_value(
                "n_params", len(self.param_vals), t=self.vqe_it
            )

            super().run()
            self.optimizer.reset()

            # seperately log the energy after each adapt iteration
            self.logger.add_logged_value(
                "energy_adapt", self.logger.logged_values["energy"][-1]
            )

            self.adapt_vqe_it += 1

        if created_pbar:
            self.progress_bar.close()
