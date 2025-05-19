from abc import ABC
from abc import abstractmethod

import torch
from tqdm import tqdm

from torchsmith.models.flow.base import ODE
from torchsmith.models.flow.base import SDE


class Solver(ABC):
    @abstractmethod
    def step(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Returns :math:`x_{t + h}` given :math:`x_t`, :math:`t` and :math:`h`."""
        raise NotImplementedError()

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Returns the final state (at ts[-1]) by updating initial state `x` through
        timesteps ts[0] to ts[-1].
        """
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(
        self, x: torch.Tensor, ts: torch.Tensor
    ) -> torch.Tensor:
        """Returns the trajectory simulated by updating initial state `x` through
        timesteps ts[0] to ts[-1].
        """
        xs = [x.clone()]  # TODO: create a tensor and preallocate
        for t_idx in tqdm(range(len(ts) - 1)):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class EulerSolver(Solver):
    """Euler method solver for ODEs by discretization.

    .. math::
        dX_t = u_t(X_t) dt  \\quad \\rightarrow \\quad X_{t + h} = X_t + hu_t(X_t)

    where :math:`h = \Delta t` is the step size.
    """

    def __init__(self, ode: ODE) -> None:
        self.ode = ode

    def step(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        u_t = self.ode.drift_coefficient(x_t, t)
        x_t_plus_h = x_t + h * u_t
        return x_t_plus_h


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
