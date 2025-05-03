from enum import Enum

from typing_extensions import Any, Self

from .adam import Adam
from .lbfgs import LBFGS
from .levenberg_marquardt import LevenbergMarquardt
from .optimizer import Optimizer
from .sgd import SGD


# TODO: decide if we actually want to make this a cmd line tool, delete this file if not
class AvailableOptimizer(str, Enum):
    SGD = "sgd"
    Adam = "adam"
    LBFGS = "lbfgs"
    LevenbergMarquardt = "levenberg_marquardt"

    def construct(self: Self, kwargs: dict[str, Any]) -> Optimizer:
        """
        Constructs the associated optimizer with the kwargs
        """

        match self:
            case AvailableOptimizer.SGD:
                return SGD(**kwargs)
            case AvailableOptimizer.Adam:
                return Adam(**kwargs)
            case AvailableOptimizer.LBFGS:
                return LBFGS(**kwargs)
            case AvailableOptimizer.LevenbergMarquardt:
                return LevenbergMarquardt(**kwargs)
            case _:
                raise NotImplementedError()
