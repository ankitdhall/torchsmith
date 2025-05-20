import torch

from torchsmith.models.flow.processes import ODE
from torchsmith.models.flow.solvers.base import Solver


class EulerSolver(Solver):
    """Euler method solver for ODEs by discretization.

    .. math::
        dX_t = u_t(X_t) dt  \\quad \\rightarrow \\quad X_{t + h} = X_t + hu_t(X_t)

    where :math:`h = \\Delta t` is the step size.
    """

    def __init__(self, ode: ODE) -> None:
        self.ode = ode

    def step(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        u_t = self.ode.drift_coefficient(x_t, t)
        x_t_plus_h = x_t + h * u_t
        return x_t_plus_h
