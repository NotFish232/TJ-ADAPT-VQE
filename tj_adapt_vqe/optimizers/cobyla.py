import nlopt # type: ignore
from typing import Callable, Optional 
import numpy as np 
from .optimizer import Optimizer

class CobylaOptimizer(Optimizer):
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        constraints: Optional[list[Callable[[np.ndarray], float]]] = None,
        tol: float = 1e-4,
        maxeval: int = 1000
    ) -> None:
        super().__init__(name="COBYLA", gradient_convergence_threshold=tol)
        self.objective_fn = objective_fn
        self.constraints = constraints or []
        self.tol = tol
        self.maxeval = maxeval
        self.last_result = None

    def reset(self) -> None:
        self.last_result = None

    def update(self, param_vals: np.ndarray, gradient: np.ndarray = None) -> np.ndarray:
        dim = len(param_vals)
        opt = nlopt.opt(nlopt.LN_COBYLA, dim)
        opt.set_min_objective(lambda x, _: self.objective_fn(np.array(x)))

        for constraint_fn in self.constraints:
            # nlopt expects inequality constraints: constraint(x) >= 0
            opt.add_inequality_constraint(lambda x, grad: constraint_fn(np.array(x)), tol=1e-8)

        opt.set_xtol_rel(self.tol)
        opt.set_maxeval(self.maxeval)

        try:
            result = opt.optimize(param_vals)
            self.last_result = (result, opt.last_optimum_value(), opt.last_optimize_result())
        except Exception as e:
            self.last_result = ("failure", str(e), None)
            return param_vals  # fallback to original if error

        return np.array(result)

    def is_converged(self, gradient: np.ndarray = None) -> bool:
        if self.last_result is None:
            return False
        result_code = self.last_result[2]
        return result_code in [nlopt.SUCCESS, nlopt.STOPVAL_REACHED, nlopt.FTOL_REACHED, nlopt.XTOL_REACHED]

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tol": self.tol,
            "maxeval": self.maxeval,
            "num_constraints": len(self.constraints),
        }
