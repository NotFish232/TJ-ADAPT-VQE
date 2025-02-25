from typing_extensions import Self

class GradientDescent(Optimizer):
    """
    Performs gradient descent to optimize circuit parameters.
    """
    def __init__(self: Self, step=0.01) -> None:
        this.step = step

    @override
    def update(self: Self) -> None:
        pass