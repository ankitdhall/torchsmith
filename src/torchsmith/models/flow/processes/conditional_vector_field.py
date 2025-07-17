import torch

from torchsmith.models.flow.paths.conditional_probability_path import (
    ConditionalProbabilityPath,
)
from torchsmith.models.flow.processes import ODE
from torchsmith.models.flow.processes import SDE


class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor) -> None:
        super().__init__()
        self.path = path
        self.z = z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns the conditional vector field u_t(x|z)."""
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x, z, t)


class ConditionalVectorFieldSDE(SDE):
    def __init__(
        self, path: ConditionalProbabilityPath, z: torch.Tensor, *, sigma: float
    ) -> None:
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns the conditional vector field u_t(x|z)."""
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(
            x, z, t
        ) + 0.5 * self.sigma**2 * self.path.conditional_score(x, z, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns the conditional vector field :math:`\\nabla log p_{density}(x|z)`."""
        return self.sigma * torch.randn_like(x)
