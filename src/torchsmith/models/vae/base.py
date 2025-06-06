from abc import ABC
from abc import abstractmethod

import numpy as np
import torch

from torchsmith.models.vae.dtypes import Loss


def reparameterize(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return mu + torch.exp(log_std) * torch.randn_like(mu)


class BaseVAE(ABC, torch.nn.Module):
    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def sample(self, num_samples: int, add_noise: bool) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def loss(self, x: torch.Tensor) -> Loss:
        # Returns: negative ELBO, reconstruction loss, KL-div loss
        raise NotImplementedError()

    @abstractmethod
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class BaseVQVAE(ABC, torch.nn.Module):
    @abstractmethod
    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def decode_from_indices(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        raise NotImplementedError()
