import numpy as np
import torch

from torchsmith.models.diffusion import DiffusionModel
from torchsmith.models.external._vae import VAE
from torchsmith.training.utils import plot_samples
from torchsmith.utils.pytorch import get_device

device = get_device()


def generate_samples_fn_2d(
    model: DiffusionModel,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    num_samples: int = 2000,
) -> np.ndarray:
    sample_steps = np.power(2, np.linspace(0, 9, 9)).astype(int)
    samples_list = []
    for num_steps in sample_steps:
        samples_for_steps = model.sample(num_samples, num_steps=num_steps)
        samples_list.append(samples_for_steps.detach().cpu().numpy())
    samples = np.array(samples_list)
    samples = (samples * std) + mean
    return samples


def generate_samples_fn_cifar_10(
    model: DiffusionModel,
    *,
    num_samples: int = 10,
    mean: float = 0.5,
    std: float = 0.5,
    sample_steps: np.ndarray | None = None,
) -> np.ndarray:
    sample_steps = (
        np.power(2, np.linspace(0, 9, 10)).astype(int)
        if sample_steps is None
        else sample_steps
    )
    samples_list = []
    for num_steps in sample_steps:
        samples_for_steps = model.sample(
            num_samples, num_steps=num_steps, clamp_to=(-1, 1)
        )
        samples_list.append(samples_for_steps.detach().cpu().numpy())

    samples = np.array(samples_list).transpose(0, 1, 3, 4, 2)  # (10, 10, 32, 32, 3)
    print(f"Generated samples with shape: {samples.shape}")
    samples = (samples * std) + mean
    _samples = samples.reshape(-1, *samples.shape[2:]).transpose(
        0, 3, 1, 2
    )  # (10*10, 3, 32, 32)
    plot_samples(255 * _samples, num_rows=num_samples, show=True)

    return samples


def generate_samples_fn_latent_cifar_10(
    model: DiffusionModel,
    *,
    vae: VAE,
    class_indices: list[int],
    scale_factor: float,
    num_samples: int = 10,
    mean: float = 0.5,
    std: float = 0.5,
    num_steps: int = 512,
    cfg_weight: float | None = None,
) -> np.ndarray:
    samples_list = []
    for class_index in class_indices:
        samples_for_steps = model.sample(
            num_samples,
            num_steps=num_steps,
            clamp_to=None,
            y=torch.full(
                (num_samples,),
                fill_value=class_index,
                device=device,
                dtype=torch.long,
            ),
            cfg_weight=cfg_weight,
        )
        samples_np = samples_for_steps.detach().cpu().numpy() * scale_factor
        samples_np_decoded = (
            vae.decode(samples_np).cpu().numpy().transpose(0, 2, 3, 1)
        )  # (B, C, H, W) -> # (B, H, W, C)
        samples_list.append(samples_np_decoded)

    samples = np.array(samples_list)  # (num_classes, B, H, W, C)
    samples = (samples * std) + mean
    print(f"Generated samples with shape: {samples.shape}")

    _samples = samples.reshape(-1, *samples.shape[2:]).transpose(
        0, 3, 1, 2
    )  # (10*10, 3, 32, 32)
    plot_samples(255 * _samples, num_rows=num_samples, show=True)

    return samples
