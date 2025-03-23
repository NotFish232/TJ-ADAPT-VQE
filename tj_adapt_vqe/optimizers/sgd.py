from typing_extensions import Self, override

from tj_adapt_vqe.optimizers.optimizer import Optimizer


class SGDOptimizer(Optimizer):
    """
    Implementation of standard SGD Gradient Descent
    """

    def __init__(self: Self) -> None:
        super().__init__()

    @override
    def update(self: Self) -> None:
        return
