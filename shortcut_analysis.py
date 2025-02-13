from adaptvqe.molecules import create_lih
from adaptvqe.pools import QE, DVE_CEO, NoZPauliPool
from adaptvqe.matrix_tools import string_to_matrix, create_unitary
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
import numpy as np
from openfermion import __str__, get_sparse_operator, FermionOperator
from scipy.sparse import csc_matrix
from adaptvqe.chemistry import get_hf_det
from adaptvqe.matrix_tools import ket_to_vector
import matplotlib.pyplot as plt

r = 2
molecule = create_lih(r)
pool = NoZPauliPool(molecule)
qubit_num = molecule.n_qubits

import json #load in data
with open(str(pool.name) + "_" + str(molecule.description) + "_r=" + str(r) + ".json") as f:
    data = json.load(f)

ref_state = ket_to_vector( get_hf_det(molecule.n_electrons, molecule.n_qubits) ) #making reference state 
ansatze_for_each_step = []

for oplist in data["evolution_of_ansatz_indices"]: #for operator list in each step
    step_num = data["evolution_of_ansatz_indices"].index(oplist) #0, 1, 2
    step_operators=[]
    step_coefficients = data["evolution_of_ansatz_coefficients"][step_num]

    for item in oplist: #for each operator in each array of operators for each step
        step_operators.append( get_sparse_operator( pool.get_op(item), n_qubits = qubit_num ) )
    
    ansatze_for_each_step.append ( create_unitary( step_coefficients, step_operators, 2**qubit_num ) )

#set up number operator
n = sum([FermionOperator(f'{i}^ {i}') for i in range(molecule.n_qubits)])
n_matrix = get_sparse_operator(n).todense()

#set up sz and total spin squared operators
sz = 0
s_plus = 0
s_minus = 0

for orb in range(int(molecule.n_qubits/2)):

    up_orb = orb*2
    down_orb = orb*2 + 1

    s_plus += FermionOperator(f'{up_orb}^ {down_orb}')
    s_minus += FermionOperator(f'{down_orb}^ {up_orb}')

    sz+= FermionOperator(f"{up_orb}^ {up_orb}",1/2)
    sz+= FermionOperator(f"{down_orb}^ {down_orb}",-1/2)

sz_matrix = get_sparse_operator(sz).todense()

s2 = s_minus*s_plus + sz*(sz+1)
s2_matrix = get_sparse_operator(s2).todense()


n_expec_values = []
sz_expec_values = []
s2_expec_values = []

step = 0
psi = 0
for ansatze in ansatze_for_each_step:
    psi = ansatze * ref_state
    step+=1
    #expectation value
    n_expec = ((psi.transpose().conj()).dot(n_matrix)).dot(psi)[0,0]
    sz_expec = ((psi.transpose().conj()).dot(sz_matrix)).dot(psi)[0,0]
    s2_expec = ((psi.transpose().conj()).dot(s2_matrix)).dot(psi)[0,0]
    
    n_expec_values.append(n_expec)
    sz_expec_values.append(sz_expec)
    s2_expec_values.append(s2_expec)


x = np.linspace(1,step,step)

plt.xticks(np.arange(step))
plt.plot(x, n_expec_values, 'red', label= str(pool.name) + ", " + str(molecule.description) + ", r = "+str(r))
plt.plot(x, sz_expec_values, 'blue', label= str(pool.name) + ", " + str(molecule.description) + ", r = "+str(r))
plt.plot(x, s2_expec_values, 'green', label= str(pool.name) + ", " + str(molecule.description) + ", r = "+str(r))
plt.xlabel('ADAPT Iteration #')
plt.ylabel('Expectation Value of Operators')
plt.title('ADAPT Iteration # vs. Expectation Value of Operators')
plt.legend(loc = 'best')
plt.savefig("PLOT: " + str(pool.name) + "_" + molecule.description + "_r=" + str(r) + ".png")