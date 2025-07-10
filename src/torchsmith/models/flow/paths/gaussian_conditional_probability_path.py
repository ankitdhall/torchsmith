import torch

from torchsmith.models.flow.data import Gaussian
from torchsmith.models.flow.data import Sampleable
from torchsmith.models.flow.paths.conditional_probability_path import (
    ConditionalProbabilityPath,
)
from torchsmith.models.flow.paths.schedulers import Alpha
from torchsmith.models.flow.paths.schedulers import Beta


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(
        self, p_source: Sampleable | None, p_data: Sampleable, alpha: Alpha, beta: Beta
    ) -> None:
        p_source = p_source or Gaussian.isotropic(p_data.num_dims, 1.0)
        super().__init__(p_source, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """Samples the conditioning variable z ~ p_data(x). Here, z = x_0."""
        return self.p_target.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Samples from the conditional distribution
        p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d). Here, z = x_0.

        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        epsilon = torch.randn_like(z)
        samples = self.alpha(t) * z + self.beta(t) * epsilon
        return samples

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the conditional vector field u_t(x|z).
        Note: Only defined on t in [0,1).

        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """
        alpha = self.alpha(t)
        alpha_dot = self.alpha.dt(t)
        beta = self.beta(t)
        beta_dot = self.beta.dt(t)
        print("alpha,alpha_dot,beta,beta_dot,")

        print(alpha.shape, alpha_dot.shape, beta.shape, beta_dot.shape)

        print("x.shape, z.shape: ", x.shape, z.shape)

        return (alpha_dot - alpha * beta_dot / beta) * z + x * beta_dot / beta

    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Note: Only defined on t in [0,1).

        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_score: conditional score (num_samples, dim)
        """
        alpha = self.alpha(t)
        beta = self.beta(t)
        return (alpha * z - x) / beta**2
