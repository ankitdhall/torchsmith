import numpy as np
import torch
import torch.distributions as D

from torchsmith.models.flow.data import Density
from torchsmith.models.flow.data import Sampleable


class GaussianMixture(torch.nn.Module, Sampleable, Density):
    def __init__(
        self,
        means: torch.Tensor,  # num_modes x data_dim
        covariances: torch.Tensor,  # num_modes x data_dim x data_dim
        weights: torch.Tensor,  # num_modes
    ) -> None:
        super().__init__()
        self.num_modes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covariances", covariances)
        self.register_buffer("weights", weights)
        self.distribution = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covariances,
                validate_args=False,
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2d(
        cls, num_modes: int, *, std: float, scale: float = 10.0, seed: int = 0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(num_modes, 2) - 0.5) * scale
        covariances = torch.diag_embed(torch.ones(num_modes, 2)) * std**2
        weights = torch.ones(num_modes)
        return cls(means, covariances, weights)

    @classmethod
    def symmetric_2d(
        cls, num_modes: int, *, std: float, scale: float = 10.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, num_modes + 1)[:num_modes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covariances = torch.diag_embed(torch.ones(num_modes, 2) * std**2)
        weights = torch.ones(num_modes) / num_modes
        return cls(means, covariances, weights)
