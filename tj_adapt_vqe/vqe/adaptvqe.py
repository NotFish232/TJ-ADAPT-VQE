import numpy as np
from openfermion import MolecularData
from typing_extensions import Self

from .vqe import VQE
from ..optimizers import Optimizer
from ..pools import Pool
from ..utils import exact_expectation_value, Measure
from ..observables import SparsePauliObservable

from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate


class ADAPTVQE(VQE):
    """
    Class implementing the ADAPT-VQE algorithm
    """

    def __init__(self: Self, molecule: MolecularData, pool: Pool, optimizer: Optimizer, num_shots: int = 1024) -> None:
        """
        Initializes the ADAPTVQE object
        Arguments:
            molecule (MolecularData): the molecular data that is used for the adapt vqe
            pool (Pool): the pool that the ADAPTVQE uses to form the Ansatz
        """
        self.pool = pool
        super().__init__(molecule, optimizer, num_shots)
        self._calculate_commutators()

    def _calculate_commutators(self: Self) -> None:
        H = self.hamiltonian.operator_qiskit
        self.commutators = [SparsePauliObservable((1j*(H @ A - A @ H).simplify()).simplify(), 'PoolCommutator', self.n_qubits) for A in self.pool.operators]

    def run(self: Self) -> None:
        iteration = 1

        self.param_vals = []
        while True:
            max_gradient = -1
            max_op = None
            op_index = -1
            for j, commutator in enumerate(self.commutators):
                obs = commutator
                gradient = Measure(self.circuit, self.param_vals, [obs], [], num_shots=self.num_shots).evs[obs]
                if abs(gradient) > max_gradient:
                    max_gradient = gradient
                    max_op = self.pool.operators[j]
                    op_index = j

            print(f"ADAPT VQE | max_gradient={max_gradient}")
            if abs(max_gradient) < 0.04: # chosen by trial and error
                break

            self.param_vals = np.append(self.param_vals, np.random.rand(1).astype(np.float32) - 0.5)
            param = Parameter(f'n{iteration}{self.pool.labels[op_index]}')
            self.circuit.compose(PauliEvolutionGate(max_op, param), inplace=True)

            self.circuit = self.circuit.decompose(reps=2)
            self.optimize_parameters()

            ev = Measure(self.circuit, self.param_vals, [self.hamiltonian], [], num_shots=self.num_shots).evs[self.hamiltonian]
            print(f"ADAPT VQE | Iteration: {iteration} | energy={ev:.5f}, param_vals={self.param_vals}")
            iteration += 1


        full_circuit = self.circuit.assign_parameters(
            {p: val for p, val in zip(self.circuit.parameters, self.param_vals)}
        )

        state_ev = exact_expectation_value(
            full_circuit, self.hamiltonian.operator_sparse
        )

        print(
            f"ADAPT VQE[HF energy: {self.molecule.hf_energy},",
            f"Exact energy: {self.molecule.fci_energy},",
            f"Calculated energy: {state_ev},",
            f"Accuracy: {state_ev / self.molecule.fci_energy:.5%}]",
        )