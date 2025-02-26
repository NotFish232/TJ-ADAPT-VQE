from typing_extensions import Self, override

from tj_adapt_vqe.optimizers.optimizer import Optimizer


class GradientDescent(Optimizer):
    """
    Performs gradient descent to optimize circuit parameters.
    """

    def __init__(self: Self, step=0.01) -> None:
        self.step = step

    @override
    def update(self: Self) -> None:
        pass
