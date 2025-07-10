from abc import ABC
from abc import abstractmethod

import torch
from torch.func import jacrev
from torch.func import vmap


class Alpha(ABC):
    def __init__(self) -> None:
        # Check alpha_t(0) = 0 and alpha_t(1) = 1
        assert torch.allclose(self(torch.zeros(1, 1)), torch.zeros(1, 1))
        assert torch.allclose(self(torch.ones(1, 1)), torch.ones(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Returns alpha_t. Should satisfy: alpha_t(0) = 0 and alpha_t(1) = 1."""
        raise NotImplementedError()

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Returns d/dt alpha_t."""
        t = t.unsqueeze(1)  # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t)  # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)  # (num_samples, 1)


class Beta(ABC):
    def __init__(self) -> None:
        # Check beta_t(0) = 1 and beta_t(1) = 0.
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1))
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Returns alpha_t. Should satisfy: beta_t(0) = 1 and beta_t(1) = 0."""
        raise NotImplementedError()

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Returns d/dt beta_t."""
        t = t.unsqueeze(1)  # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t)  # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)  # (num_samples, 1)


class LinearAlpha(Alpha):
    """Implements alpha_t = t."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Returns alpha_t = t."""
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Returns d/dt alpha_t. Here, d/dt alpha_t = 1."""
        return torch.ones_like(t)


class SquareRootBeta(Beta):
    """Implements beta_t = sqrt(1-t)"""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Returns beta_t = sqrt(1-t)."""
        return torch.sqrt(1 - t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Returns d/dt beta_t. Here d/dt beta_t = -0.5/sqrt(1-t)."""
        return -0.5 / (torch.sqrt(1 - t) + 1e-4)
