import torch
import torch.distributions as D

from torchsmith.models.flow.data.base import Density
from torchsmith.models.flow.data.base import Sampleable


class Gaussian(torch.nn.Module, Sampleable, Density):
    def __init__(self, mean: torch.Tensor, *, cov: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
        self.distribution = D.MultivariateNormal(self.mean, self.cov)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)
