import torch

from torchsmith.models.flow.processes import SDE
from torchsmith.models.flow.solvers.base import Solver


class EulerMaruyamaSolver(Solver):
    """Euler Maruyama method solver for SDEs by discretization.

    .. math::
        dX_t = u(X_t,t) dt + \\sigma_t d W_t  \\quad \\rightarrow \\quad
        X_{t + h} = X_t + hu_t(X_t) + \\sqrt{h} \\sigma_t z_t
        \\quad z_t \\sim N(0, I_d)
    """

    def __init__(self, sde: SDE) -> None:
        self.sde = sde

    def step(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        u_t = self.sde.drift_coefficient(x_t, t)
        sigma_t = self.sde.diffusion_coefficient(x_t, t)
        z_t = torch.normal(0, 1, size=x_t.shape)
        x_t_plus_h = x_t + h * u_t + torch.sqrt(h) * sigma_t * z_t
        return x_t_plus_h
