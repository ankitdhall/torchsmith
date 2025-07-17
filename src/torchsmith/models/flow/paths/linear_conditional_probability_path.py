import torch

from torchsmith.models.flow.data import Sampleable
from torchsmith.models.flow.paths.conditional_probability_path import (
    ConditionalProbabilityPath,
)


class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_source: Sampleable, p_target: Sampleable):
        super().__init__(p_source, p_target)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """Samples the conditioning variable z ~ p_data(x). Here, z = x_0."""
        return self.p_target.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Samples from the conditional distribution
        X_t ~ p_t(x|z) such that X_t = (1-t) * X_0 + t * z.
        Here, X_0 ~ p_source.
        Here, z ~ p_target.

        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        num_samples = z.shape[0]
        x_0 = self.p_source.sample(num_samples)
        return (1 - t) * x_0 + t * z

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the conditional vector field u_t(x|z) = (z - x) / (1 - t)
        Note: Only defined on t in [0,1)

        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """
        return (z - x) / (1 - t)

    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Not defined for Linear Conditional Probability Paths."""
        raise Exception("Not defined for Linear Conditional Probability Paths.")
