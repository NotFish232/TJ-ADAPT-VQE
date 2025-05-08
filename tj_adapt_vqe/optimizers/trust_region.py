import numpy as np
from typing_extensions import Callable, Self, override

from .optimizer import HybridOptimizer


class TrustRegion(HybridOptimizer):
    """
    Trust Region optimizer using BFGS Hessian approximation
    """

    def __init__(
        self: Self,
        initial_radius: float = 1.0,
        max_radius: float = 100.0,
        min_radius: float = 1e-6,
        acceptance_threshold: float = 0.1,
        cg_tol: float = 1e-6
    ) -> None:
        super().__init__("trust_region_optimizer")
        self.delta = initial_radius
        self.max_delta = max_radius
        self.min_delta = min_radius
        self.eta = acceptance_threshold
        self.tol = cg_tol
        
        self.hessian = None
        self.prev_grad = None
        self.prev_step = None

    def _solve_subproblem(self, g: np.ndarray, B: np.ndarray, delta: float) -> np.ndarray:
        """Steihaug's conjugate gradient method for trust region subproblem"""
        p = np.zeros_like(g)
        r = g.copy()
        d = -r
        
        for _ in range(g.size):
            Bd = B @ d
            dBd = np.dot(d, Bd)
            
            if dBd <= 0:
                a = np.dot(d, d)
                b = 2 * np.dot(p, d)
                c = np.dot(p, p) - delta**2
                root = (-b + np.sqrt(max(b**2 - 4*a*c, 0))) / (2*a)
                return p + root * d
            
            alpha = np.dot(r, r) / dBd
            p_next = p + alpha * d
            
            if np.linalg.norm(p_next) >= delta:
                return (delta / np.linalg.norm(p_next)) * p_next
                
            r_next = r + alpha * Bd
            beta = np.dot(r_next, r_next) / np.dot(r, r)
            d = -r_next + beta * d
            p, r = p_next, r_next
            
            if np.linalg.norm(r) < self.tol:
                break
                
        return p

    @override
    def update(self: Self, params: np.ndarray, gradient: np.ndarray, f: Callable[[np.ndarray], float]) -> np.ndarray:
        # f_old = measure._calculate_expectation_value()
        # g = measure.gradients.copy()
        
        # if self.hessian is None:
        #     self.hessian = np.eye(params.size)
            
        # step = self._solve_subproblem(g, self.hessian, self.delta)
        
        # original_params = measure.param_values.copy()
        # measure.param_values = params + step
        # f_new = measure._calculate_expectation_value()
        # measure.param_values = original_params
        
        # #rho is reduction ratio
        # actual_reduction = f_old - f_new
        # model_reduction = -(np.dot(g, step) + 0.5 * step @ self.hessian @ step)
        # rho = actual_reduction / model_reduction if model_reduction > 1e-12 else 0.0
        
        # #here update trust region radius
        # if rho < 0.25:
        #     self.delta = max(self.delta * 0.25, self.min_delta)
        # elif rho > 0.75 and np.linalg.norm(step) > 0.8 * self.delta:
        #     self.delta = min(self.delta * 2.0, self.max_delta)
        
        # #if enough improvement then update step
        # if rho > self.eta:
        #     params += step
        #     #BFGS update if step accepted
        #     if self.prev_grad is not None:
        #         y = g - self.prev_grad
        #         s = step
        #         sy = np.dot(s, y)
                
        #         if sy > 1e-10:
        #             Bs = self.hessian @ s
        #             self.hessian += np.outer(y, y)/sy - np.outer(Bs, Bs)/np.dot(s, Bs)
            
        #     self.prev_grad = g.copy()
            
        return params