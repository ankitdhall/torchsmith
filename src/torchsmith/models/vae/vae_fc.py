import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence

from torchsmith.models.vae.base import BaseVAE
from torchsmith.models.vae.base import reparameterize
from torchsmith.utils.pytorch import add_save_load
from torchsmith.utils.pytorch import get_device

device = get_device()


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...] | int,
        *,
        output_shape: tuple[int, ...] | int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.input_dim = np.prod(input_shape).item()
        self.output_dim = np.prod(output_shape).item()
        self.output_shape = (
            (output_shape,) if isinstance(output_shape, int) else output_shape
        )
        layers = []
        previous_dim = self.input_dim
        for current_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, current_dim))
            layers.append(nn.ReLU())
            previous_dim = current_dim
        layers.append(nn.Linear(previous_dim, self.output_dim))
        print(f"Created MLP with {layers}")
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.input_dim)
        output = self.model(x)
        return output.reshape(-1, *self.output_shape)


@add_save_load
class VAE1D(BaseVAE):
    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int,
        encoder_hidden_dims: list[int],
        decoder_hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.encoder = MLP(
            input_shape=input_dim,
            output_shape=2 * latent_dim,
            hidden_dims=encoder_hidden_dims,
        )
        self.decoder = MLP(
            input_shape=latent_dim,
            output_shape=2 * input_dim,
            hidden_dims=decoder_hidden_dims,
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_z, log_std_z = self.encoder(x).chunk(2, dim=1)  # (B, N_i) -> (B, 2*N_l)
        return mu_z, log_std_z

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_x, log_std_x = self.decoder(z).chunk(2, dim=1)  # (B, N_l) -> (B, 2*N_i)
        return mu_x, log_std_x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, log_std_z = self.encode(x)
        z = reparameterize(mu_z, log_std_z)
        mu_x, log_std_x = self.decode(z)
        return mu_z, log_std_z, mu_x, log_std_x

    @torch.no_grad()
    def sample(self, num_samples: int, add_noise: bool) -> np.ndarray:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        mu_x, log_std_x = self.decode(z)
        x = reparameterize(mu_x, log_std_x) if add_noise else mu_x
        return x.cpu().numpy()

    def loss(self, x: torch.Tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        mu_z, log_std_z, mu_x, log_std_x = self(x)
        loss, loss_reconstruction, loss_kl_div = loss_function(
            x=x, mu_z=mu_z, log_std_z=log_std_z, mu_x=mu_x, log_std_x=log_std_x
        )
        return loss, loss_reconstruction, loss_kl_div


def negative_log_p_normal_distribution(
    x: torch.tensor, *, mu_x: torch.tensor, log_std_x: torch.tensor
) -> torch.tensor:
    # Use np.pi and np.log to compute constants.
    return (
        0.5 * np.log(2 * np.pi)
        + log_std_x
        + (x - mu_x) ** 2 * (1 / (2 * torch.exp(2 * log_std_x)))
    )


def loss_function(
    *,
    x: torch.tensor,
    mu_x: torch.tensor,
    log_std_x: torch.tensor,
    mu_z: torch.tensor,
    log_std_z: torch.tensor,
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    loss_reconstruction = (
        negative_log_p_normal_distribution(x, mu_x=mu_x, log_std_x=log_std_x)
        .sum(1)
        .sum()
    )

    # Here, p(z) = N(0, 1)
    random_variable_p_z = Normal(
        torch.full_like(mu_z, 0), torch.full_like(log_std_z, 1)
    )
    # Here, q_phi(z | x) = N(mu_z, exp(log_std_z))
    random_variable_q_phi_z_given_x = Normal(mu_z, torch.exp(log_std_z))
    loss_kl_div = (
        kl_divergence(random_variable_q_phi_z_given_x, random_variable_p_z).sum(1).sum()
    )

    return loss_reconstruction + loss_kl_div, loss_reconstruction, loss_kl_div
