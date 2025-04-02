
from typing_extensions import Self


class Observable:
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

class SpinZ(Observable):
    def __init__(self: Self) -> None:
        super().__init__("Spin Z")

    def expectation_value(self, state):
        val = "state* x operator x state  "
        return val

class SpinSquared(Observable):
    def __init__(self: Self) -> None:
        super().__init__("Spin Squared")

    def expectation_value(self: Self, state):
        val = "basically method used for spinZ but make the operator spin z^2+spin y^2+spin x^2"
        return val
    
class NumberOperator(Observable):
    def __init__(self: Self) -> None:
        super().__init__("Number Operator")
    
    def expectation_value(self: Self, state):
        val = "yk"
        return val