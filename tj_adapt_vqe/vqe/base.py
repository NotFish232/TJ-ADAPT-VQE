from typing_extensions import Self


class VQE:
    """
    Class for the actual variational quantum eigensolvers algorithm
    """

    def __init__(self: Self) -> None:
        """
        Initializes starting Ansatze (constructor probably needs to take num qubits?)
        Maybe take a callback or something if we want a better starting point
        Also need either molecule / moleculear hamiltonian to actually calculate
        the expected value on our Ansatze (william says molecule is nice please supply that)
        """
        


    def train(self: Self) -> None:
        """
        trains the VQE using a pool (pool might be saved in the constructor or passed here)
        while training:
            # probably make a method for gradient calculation
            train()
            if self._update_ansatz is not None:
                self._update_ansatz()
        """
        
