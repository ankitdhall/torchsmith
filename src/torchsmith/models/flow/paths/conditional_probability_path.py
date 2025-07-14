from abc import ABC
from abc import abstractmethod

import torch

from torchsmith.models.flow.data import Sampleable


class ConditionalProbabilityPath(torch.nn.Module, ABC):
    def __init__(self, p_source: Sampleable, p_target: Sampleable) -> None:
        super().__init__()
        self.p_source = p_source
        self.p_target = p_target

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)"""
        assert t.dim() == 2 and t.shape[1] == 1, "t must be of shape (num_samples, 1)"
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z = self.sample_conditioning_variable(num_samples)  # (num_samples, dim)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t)  # (num_samples, dim)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """Samples the conditioning variable z from p(z)."""
        raise NotImplementedError()

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Samples from the conditional distribution p_t(x|z) given z at time t."""
        raise NotImplementedError()

    @abstractmethod
    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Returns the conditional vector field u_t(x|z) for x given z at time t."""
        raise NotImplementedError()

    @abstractmethod
    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Returns the conditional score of p_t(x|z) for x given z at time t."""
        raise NotImplementedError()
