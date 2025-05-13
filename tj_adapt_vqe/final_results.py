'''
Create graphs of the following; use the logger to display them:

for each pool...
1) Iteration vs. particle number
2) Iteration vs. spin z 
3) Iteration vs. total spin squared 
'''

from utils import logger import Logger 

logger = Logger()

logger.add_config_option("pool", "Qubit + QE")

def log_gradients_from_file(filename: str, metric_name: str):
    with open(filename, "r") as f:
        for line in f:
            i, g = line.strip().split(",")
            i = int(i)
            g = float(g)
            logger.add_logged_value(name=metric_name, value=g, t=i)

# Log both sets
log_gradients_from_file("gradients_QE.txt", "max_gradient_qubit")
log_gradients_from_file("gradients_combined.txt", "max_gradient_combined")
log_gradients_from_file("particle_number_QE_and_Qubit.txt", "particle_number_combined")
log_gradients_from_file("spin_z_QE_and_Qubit.txt", "spin_z_combined")
log_gradients_from_file("total_spin_squared_QE_and_Qubit.txt", "total_spin_squared_combined")
