from enum import Enum

import numpy as np
from openfermion import MolecularData
from qiskit.circuit import Gate, QuantumCircuit  # type: ignore
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Self, override

from ..observables import Observable, SparsePauliObservable
from ..observables.measure import Measure
from ..optimizers import Optimizer
from ..pools import Pool
from ..utils.ansatz import make_perfect_pair_ansatz
from ..utils.conversions import prepend_params
from .vqe import VQE


class ADAPTConvergenceCriteria(Enum):
    """
    Inherits from `enum.Enum`. An enum representing the different convergence criteria for the ADAPT algorithm.

    Members:
        `ADAPTConvergenceCriteria.Gradient` conoverged if gradient of selected operator falls below a threshold.
        `ADAPTConvergenceCriteria.LackOfImprovement` converged if change in energy betwen iterations falls below a threshold.
    """

    Gradient = 0
    LackOfImprovement = 1


class ADAPTVQE(VQE):
    """
    Inherits from `VQE`. Implements the ADAPT VQE algorithm which selects new operators for the ansatz
    based on the operator with the highest gradient within the pool.
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
        Constructs an instance of `ADAPTVQE`. Calls `super().__init__()` with the passed
        `molecule`, `optimizer`, `observables`, and `num_shots`. Calculates the commutator
        of each operator within the passed pool and the molecular hamiltonian.

        Args:
            self (Self): A reference to the current class instance.
            molecule (MolecularData): The molecule that the ADAPTVQE algorithm will be ran on.
            pool (Pool): The pool to select operators from.
            optimizer (Optimizer): The optimizer to perform each VQE iteration on. Passed to super class.
            observables (list[Observable], optional): The observables to track. Passed to super class. Defaults to [].
            num_shots (int, optional): The num shots for simulations. Passed to super class. Defaults to 1024.
            adapt_conv_criteria (ADAPTConvergenceCriteria, optional): The criteria to use for ADAPT convergence. Defaults to ADAPTConvergenceCriteria.Gradient.
            conv_threshold (float, optional): The threshold that the criteria uses to determine ADAPT convergence. Defaults to 0.01.
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
        Overrides the VQE implementation of `_make_ansatz(...)` to create a simpler ansatz
        that works better with ADAPTVQE and selecting operators dynamically. Can either be set
        to the perfect pair configuration or the hartfree-fock state.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            QuantumCircuit: The initial ansatz for the ADAPTVQE algorithm to use.
        """

        ansatz = make_perfect_pair_ansatz(self.n_qubits)

        return ansatz

    @override
    def _make_progress_description(self: Self) -> str:
        """
        Overrides the VQE implementation of `_make_progress_description(...)` to create a progress bar with
        more information. Includes the implementation of `super()._make_progress_description(...)` in its output.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            str: A string representing the description of the tqdm progress bar.
        """

        vqe_progress_descrption = super()._make_progress_description()

        n_params_f = f"{len(self.circuit.parameters)}"

        return f"ADAPT-VQE it: {self.adapt_vqe_it} | {vqe_progress_descrption} | N-Params: {n_params_f}"

    def _calculate_commutators(self: Self) -> tuple[list[Observable], list[int]]:
        """
        Calculates the commutator between `self.hamiltonian` and every Observable in `self.pool`.
        Also returns the "density" for each operator, which is how many matrix operators
        each qiskit operator within the pool represents. For instance, `pool.get_op(...)`
        could return an array of length 3 with 3 actual operators that get fused together in `pool.get_exp_op(...)`
        with gradients calculated seperately but only one qiskit circuit.


        Args:
            self (Self): A reference to the current class instance.

        Returns:
            tuple[list[Observable],list[int]]: A list of Observable wrapped commutators and a list of the count of each operator
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
        Finds the best operator in the pool by calculating the expectation value of each operator's commutator
        and finding the operator that has the largest absolute value of the expectation value. If an operator
        corresponds to multiple matrix operators, the sum of the absolute values of expectation values will be
        substituted for the singular expectation value.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            tuple[float,int]: A tuple of the highest gradient and its corresponding pool operator index.
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

    def _is_converged(self: Self) -> bool:
        """
        Checks whether the ADAPT-VQE algorithm is converged. Utilizes information that has been logged
        to `logger.logged_values`. Uses `self.adapt_conv_criteria` alongside `self.conv_threshold` to decide
        whether the algorithm is converged.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            bool: Whether ADAPT VQE has converged.
        """

        if self.adapt_conv_criteria == ADAPTConvergenceCriteria.Gradient:
            prev_op_grad = self.logger.logged_values["adapt_operator_grad"][-1]
            return prev_op_grad < self.conv_threshold

        if self.adapt_conv_criteria == ADAPTConvergenceCriteria.LackOfImprovement:
            if self.adapt_vqe_it <= 2:
                return False

            adapt_energies = self.logger.logged_values["adapt_energy"]

            return abs(adapt_energies[-1] - adapt_energies[-2]) < self.conv_threshold

    def _prepare_new_op(self: Self, idx: int) -> QuantumCircuit:
        """
        Prepares an operator to be added to the quantum circuit. This includes
        renaming parameters to ensure uniqueness and turning qiskit `Gate` instances
        to `QuantumCircuit` instances.

        Args:
            self (Self): A reference to the current class instance.
            idx (int) The idx of the operator in the pool that should be prepared.

        Returns:
            QuantumCircuit: The circuit representing the operator.
        """

        op = self.pool.get_exp_op(idx)
        label = self.pool.get_label(idx)

        if isinstance(op, Gate):
            op = QuantumCircuit(op.num_qubits).compose(op)
        op = prepend_params(op, f"n{self.adapt_vqe_it}{label}")

        return op

    def run(self: Self) -> None:
        """
        Runs the ADAPT VQE algorithm. Each iteration, a new operator is chosen, appened to the end
        of the ansatz, `super().run(...)` is called to optimize VQE parameters, and then repeated until convergence.
        Through this process, values are also logged to `self.logger`.

        Args:
            self (Self): A reference to the current class instance.
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

            self.logger.add_logged_value("adapt_operator_idx", max_idx)
            self.logger.add_logged_value("adapt_operator_grad", max_grad)

            # convergence checks, seems hard to seperate this into its own function
            if self._is_converged():
                break

            new_op = self._prepare_new_op(max_idx)
            self.circuit.compose(new_op, inplace=True)
            self.transpiled_circuit = self._transpile_circuit(self.circuit)

            self.param_vals = np.append(
                self.param_vals, np.zeros(len(new_op.parameters))
            )

            self.logger.add_logged_value(
                "n_params", len(self.param_vals), t=self.vqe_it
            )

            super().run()
            self.optimizer.reset()

            # seperately log the energy after each adapt iteration
            self.logger.add_logged_value(
                "adapt_energy", self.logger.logged_values["energy"][-1]
            )

            self.adapt_vqe_it += 1

        if created_pbar:
            self.progress_bar.close()
