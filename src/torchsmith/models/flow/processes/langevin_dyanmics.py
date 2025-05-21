import torch

from torchsmith.models.flow.data.base import Density
from torchsmith.models.flow.processes.base import SDE


class LangevinDynamics(SDE):
    """Langevin Dynamics.

    .. math::
        dX_t = 0.5 * \\sigma^2 \\nabla \\log p(X_t) dt + \\sigma dW_t, \\quad X_0 = x_0.

    where the score function is defined as :math:`\\nabla log p_{density}(x)`.
    """

    def __init__(self, sigma: float, density: Density):
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.sigma**2) * self.density.score(x_t)

    def diffusion_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x_t, self.sigma)
