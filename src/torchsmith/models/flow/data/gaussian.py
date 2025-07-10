import torch
import torch.distributions as D

from torchsmith.models.flow.data.base import SampleableDensity


class Gaussian(torch.nn.Module, SampleableDensity):
    def __init__(self, mean: torch.Tensor, *, cov: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
        self.distribution = D.MultivariateNormal(self.mean, self.cov)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @property
    def num_dims(self) -> int:
        return self.distribution.mean.dim()

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov=cov)
