import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence
from torch.nn import functional as F

from torchsmith.models.vae.base import BaseVAE
from torchsmith.models.vae.base import reparameterize
from torchsmith.models.vae.dtypes import VAELoss
from torchsmith.utils.pytorch import add_save_load
from torchsmith.utils.pytorch import get_device

device = get_device()


class EncoderConv(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        *,
        latent_dim: int,
    ) -> None:
        super().__init__()
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.input_channels = input_shape[0]
        self.spatial_dims = input_shape[1]
        layers = [
            torch.nn.Conv2d(self.input_channels, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),  # 16 x 16
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1),  # 8 x 8
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, 2, 1),  # 4 x 4
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                256 * self.spatial_dims // 8 * self.spatial_dims // 8, latent_dim * 2
            ),
        ]
        print(f"Created MLP with {layers}")
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_z, log_std_z = self.model(x).chunk(
            2, dim=1
        )  # (B, N_i) -> (B, N_l), (B, N_l)
        return mu_z, log_std_z


class DecoderConv(nn.Module):
    def __init__(
        self,
        output_shape: tuple[int, ...],
        *,
        latent_dim: int,
    ) -> None:
        super().__init__()
        assert len(output_shape) == 3
        assert output_shape[1] == output_shape[2]
        self.output_channels = output_shape[0]
        self.spatial_dims = output_shape[1]

        self.decoder_input_shape = (128, self.spatial_dims // 8, self.spatial_dims // 8)
        self.fc = torch.nn.Linear(latent_dim, np.prod(self.decoder_input_shape).item())
        layers = [
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, self.output_channels, 3, 1, 1),
        ]
        print(f"Created MLP with {layers}")
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc_output = self.fc(x).reshape(-1, *self.decoder_input_shape)
        return self.model(fc_output)


@add_save_load
class VAEConv(BaseVAE):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        *,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = EncoderConv(
            input_shape=input_shape,
            latent_dim=latent_dim,
        )
        self.decoder = DecoderConv(
            output_shape=input_shape,
            latent_dim=latent_dim,
        )
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, log_std_z = self.encoder(x)
        z = reparameterize(mu_z, log_std_z)
        x_reconstructed = self.decoder(z)
        return mu_z, log_std_z, x, x_reconstructed

    @torch.no_grad()
    def sample(self, num_samples: int) -> np.ndarray:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        x = self.decoder(z)
        return x.cpu().numpy()

    def loss(self, x: torch.Tensor) -> VAELoss:
        mu_z, log_std_z, x, x_reconstructed = self(x)
        loss, loss_reconstruction, loss_kl_div = loss_function_conv(
            mu_z=mu_z, log_std_z=log_std_z, x=x, x_reconstructed=x_reconstructed
        )
        return VAELoss(
            negative_ELBO=loss,
            reconstruction_loss=loss_reconstruction,
            KL_div_loss=loss_kl_div,
        )

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu_z, _ = self.encoder(x)
        x_hat = self.decoder(mu_z)
        return x_hat


def loss_function_conv(
    *,
    x: torch.tensor,
    x_reconstructed: torch.tensor,
    mu_z: torch.tensor,
    log_std_z: torch.tensor,
) -> torch.tensor:
    loss_reconstruction = F.mse_loss(x_reconstructed, x, reduction="sum")

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
