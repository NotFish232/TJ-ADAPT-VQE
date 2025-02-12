from typing_extensions import Self


class ADAPTVQE(VQE):
    """
    Class implementing the ADAPT-VQE algorithm
    """
    pool = None

    def __init__(self: Self, pool):
        """
        Initializes the ADAPTVQE object
        Arguments:
            pool (Pool): the pool that the ADAPTVQE uses to form the Ansatz
        """
        self.pool = pool

    def make_ansatz():
        pass

    def run(self: Self):
        """
        Runs the ADAPTVQE algorithm. The main loop goes like this:
        1. measure gradient
        2. if converged, stop
        3. select operator with the largest gradient
        4. grow ansatz
        5. run self.optimize
        6. goto 1
        """