from typing import Callable

import numpy as np
import torch

from torchsmith.models.vae.vae_conv import VAEConv
from torchsmith.training.utils import plot_samples
from torchsmith.utils.pytorch import get_device

device = get_device()


def generate_samples(
    model: VAEConv, *, num_samples: int = 100, postprocess_fn: Callable | None
) -> np.ndarray:
    samples = model.sample(num_samples)
    samples = postprocess_fn(samples) if postprocess_fn is not None else samples
    plot_samples(samples, num_rows=int(num_samples**0.5), show=True)
    return samples


def generate_reconstructions(
    num_samples: int,
    model: VAEConv,
    dataloader: torch.utils.data.DataLoader,
    postprocess_fn: Callable | None,
) -> np.ndarray:
    x = next(iter(dataloader))[:num_samples].to(device)
    with torch.no_grad():
        z, _ = model.encoder(x)
        x_recon = model.decoder(z)
    reconstructions = np.stack((x.cpu(), x_recon.cpu()), axis=1).reshape(
        (-1, 3, 32, 32)
    )
    reconstructions = (
        postprocess_fn(reconstructions)
        if postprocess_fn is not None
        else reconstructions
    )
    return reconstructions


def generate_interpolations(
    *,
    model: VAEConv,
    num_samples: int,
    dataloader: torch.utils.data.DataLoader,
    num_steps: int = 10,
    postprocess_fn: Callable | None,
) -> np.ndarray:
    x = next(iter(dataloader))[: 2 * num_samples].to(device)
    with torch.no_grad():
        z, _ = model.encoder(x)  # (20, N_l)
        z1, z2 = z.chunk(2, dim=0)  # (20, N_l) -> # (10, N_l), (10, N_l)
        interpolations = [
            model.decoder(z1 * (1 - alpha) + z2 * alpha).cpu()
            for alpha in np.linspace(0, 1, num_steps)
        ]
        interpolations = np.stack(interpolations, axis=1).reshape((-1, 3, 32, 32))
    interpolations = (
        postprocess_fn(interpolations) if postprocess_fn is not None else interpolations
    )
    return interpolations
