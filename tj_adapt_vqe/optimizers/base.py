from typing_extensions import Self


class Optimizer:
    """
    Base Class that all other optimizers should inherit from
    Should probably take a reference to parameters to optimize and optimize them in place
    to prevent unncessary copying
    """

    def __init__(self: Self) -> None:
        pass

    def update(self: Self):
        """
        Performs one step of update
        """
