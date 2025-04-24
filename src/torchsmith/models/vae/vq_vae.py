import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from torchsmith.models.vae.base import BaseVAE
from torchsmith.models.vae.base import BaseVQVAE
from torchsmith.models.vae.dtypes import VQVAELoss
from torchsmith.utils.pytorch import add_save_load
from torchsmith.utils.pytorch import get_device

device = get_device()


class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        layers = [
            torch.nn.BatchNorm2d(self.num_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.BatchNorm2d(self.num_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                self.num_channels, self.num_channels, kernel_size=1, stride=1, padding=0
            ),
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B, C, H, W) -> (B, C, H, W)


class Encoder(torch.nn.Module):
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
        self.latent_dim = latent_dim
        layers = [
            torch.nn.Conv2d(
                self.input_channels, self.latent_dim, kernel_size=4, stride=2, padding=1
            ),  # 1/2 x 1/2
            torch.nn.BatchNorm2d(self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                self.latent_dim, self.latent_dim, kernel_size=4, stride=2, padding=1
            ),  # 1/4 x 1/4
            ResidualBlock(self.latent_dim),
            ResidualBlock(self.latent_dim),
        ]
        self.model = torch.nn.Sequential(*layers)

    def output_shape(self) -> torch.Size:
        return torch.Size([self.spatial_dims // 4, self.spatial_dims // 4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B, C, H, W) -> (B, C, H / 4, W / 4)


class Decoder(torch.nn.Module):
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
        self.latent_dim = latent_dim
        layers = [
            ResidualBlock(self.latent_dim),
            ResidualBlock(self.latent_dim),
            torch.nn.BatchNorm2d(self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                self.latent_dim, self.latent_dim, kernel_size=4, stride=2, padding=1
            ),  # 1/4 x 1/4 -> 1/2 x 1/2
            torch.nn.BatchNorm2d(self.latent_dim),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                self.latent_dim,
                self.output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 1/2 x 1/2 -> 1x1
            torch.nn.Tanh(),
        ]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B, C, H / 4, W / 4) -> (B, C, H, W)


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        *,
        size: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.size = size
        self.latent_dim = latent_dim
        self.embeddings = torch.nn.Embedding(self.size, self.latent_dim)  # (N, D)
        self.embeddings.weight.data.uniform_(-1 / self.size, 1 / self.size)

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H, W, C)
        x = rearrange(x, "B C H W -> B H W C")

        # Compute distance between embeddings and input
        # dist((B, H, W, D), (N, D)) -> (B, H, W, N) (here, C in x is the same as D)
        distance_matrix = torch.cdist(x, self.embeddings.weight, p=2)
        # argmin((B, H, W, N), dim=-1) -> (B, H, W)
        indices_to_closest_embedding = distance_matrix.argmin(dim=-1)
        return indices_to_closest_embedding

    def decode_to_features(self, indices: torch.Tensor) -> torch.Tensor:
        # Select indices (B, H, W) from (N, D) -> (B, H, W, D)
        B, H, W = indices.shape
        indices = rearrange(indices, "B H W -> (B H W)")
        x_quantized = torch.index_select(self.embeddings.weight, dim=0, index=indices)
        return rearrange(x_quantized, "(B H W) D -> B D H W", B=B, H=H, W=W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_to_closest_embedding = self.encode_to_indices(
            x
        )  # (B, C, H, W) -> (B, H, W)

        x_quantized = self.decode_to_features(indices_to_closest_embedding)
        x_q_with_ste = (x_quantized - x).detach() + x
        return x_q_with_ste


@add_save_load
class VQVAE(BaseVAE, BaseVQVAE):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        *,
        latent_dim: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
        )
        self.codebook = VectorQuantizer(size=codebook_size, latent_dim=latent_dim)
        self.decoder = Decoder(
            output_shape=input_shape,
            latent_dim=latent_dim,
        )
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    @property
    def codebook_size(self) -> int:
        return self.codebook.size

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        indices = self.codebook.encode_to_indices(z_e)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_q = self.codebook.decode_to_features(x)
        x_hat = self.decoder(z_q)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        z_q = self.codebook(z_e)
        x_hat = self.decoder(z_q)
        return x_hat

    @torch.no_grad()
    def sample(self, num_samples: int) -> np.ndarray:
        latent_indices = torch.randint(
            low=0,
            high=self.codebook.size,
            size=(num_samples, *self.encoder.output_shape()),
            device=device,
        )
        x_hat = self.decode_from_indices(latent_indices)
        return x_hat.cpu().numpy()

    def loss(self, x: torch.Tensor) -> VQVAELoss:
        z_e = self.encoder(x)
        z_q = self.codebook(z_e)
        x_reconstructed = self.decoder(z_q)

        reconstruction_loss = (
            F.mse_loss(x_reconstructed, x, reduction="none").mean(dim=[1, 2, 3]).sum()
        )
        loss_commitment = (
            F.mse_loss(z_e, z_q.detach(), reduction="none").mean(dim=[1, 2, 3]).sum()
        )
        loss_codebook = (
            F.mse_loss(z_e.detach(), z_q, reduction="none").mean(dim=[1, 2, 3]).sum()
        )

        codebook_encoder_loss = loss_commitment + loss_codebook

        total_loss = reconstruction_loss + codebook_encoder_loss
        return VQVAELoss(
            total_loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            codebook_encoder_loss=codebook_encoder_loss,
        )

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)
