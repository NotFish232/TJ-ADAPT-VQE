
from typing_extensions import Self #justin had this in his code but i lowk dont know what it does lol never seen it b4
from openfermion import MolecularData

class Observable():
    """
    Base class for all observables
    """

    def __init__(self: Self, name) -> None:
        """
        initialize the observable to its starting value
        """
        self.name = name

    def calculate_expectation_value(self: Self) -> None:
        """
        calculates expectation value for a given observable
        """

class spinZ(Observable):
    def __init__(self):
        super().__init__("spin z")

    def expectation_value(self, state):
        val = "state* x operator x state  "
        return val

class spinSquared(Observable):
    def __init__(self):
        super().__init__("spin squared")

    def expectation_value(self, state):
        val = "basically method used for spinZ but make the operator spin z^2+spin y^2+spin x^2"
        return val
    
class numberOperator(Observable):
    def __init__(self):
        super().__init__("number op")
    
    def expectation_value(self, state):
        val = "yk"
        return val