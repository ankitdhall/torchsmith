from pathlib import Path

import huggingface_hub
import numpy as np
import pytest
import torch
from tqdm import tqdm

from torchsmith.models.diffusion import MLP
from torchsmith.models.diffusion import DiffusionModel
from torchsmith.models.diffusion import generate_samples_fn_cifar_10
from torchsmith.models.diffusion import generate_samples_fn_latent_cifar_10
from torchsmith.models.diffusion.dit import DiT
from torchsmith.models.diffusion.dit import PositionalEncoding2D
from torchsmith.models.diffusion.unet import UNet
from torchsmith.models.external.cifar_10 import load_pretrain_vqvae
from torchsmith.training.utils import plot_samples
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


def test_diffusion_sample() -> None:
    model = DiffusionModel(input_shape=2, model=MLP(input_dim=2 + 1, output_dim=2))
    x = model.sample(10, num_steps=15)
    assert x.shape == torch.Size([10, 2])


def test_unet() -> None:
    model = UNet(
        num_channels_in=3,
        num_hidden_dims=[64, 128, 256, 512],
        num_blocks_per_hidden_dim=2,
    )
    shape = (5, 3, 32, 32)
    t = torch.rand((shape[0], 1))
    x = model(torch.randn(*shape), t)
    assert x.shape == torch.Size(shape)


def test_diffusion_sample_with_unet() -> None:
    model = UNet(
        num_channels_in=3,
        num_hidden_dims=[64, 128, 256, 512],
        num_blocks_per_hidden_dim=2,
    )
    shape = (5, 3, 32, 32)
    t = torch.rand((shape[0], 1))
    x = model(torch.randn(*shape), t)
    assert x.shape == torch.Size(shape)

    diffusion_model = DiffusionModel(input_shape=(3, 32, 32), model=model)
    x = np.array(
        [
            np.zeros((10, 3, 32, 32)),
            diffusion_model.sample(10, num_steps=1),
            np.full((10, 3, 32, 32), 128),
            diffusion_model.sample(10, num_steps=2),
        ]
    )
    assert x.shape == (4, 10, 3, 32, 32)
    x = x.reshape(-1, *x.shape[2:])
    with suppress_plot():
        plot_samples(np.clip(x, 0, 255), show=True, num_rows=4)


def test_diffusion_sample_with_unet_from_loaded_model(tmp_path: Path) -> None:
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/cifar_10_latent_diffusion_UNet", filename="model.pth"
    )
    model = UNet.load_model(path_to_weights).to(device)

    diffusion_model = DiffusionModel(input_shape=(3, 32, 32), model=model)
    num_samples = 100

    for steps in tqdm([1]):  # [4, 16, 64, 256, 512]
        with suppress_plot():
            samples = generate_samples_fn_cifar_10(
                model=diffusion_model,
                num_samples=num_samples,
                sample_steps=np.array([steps]),
            )
            plot_samples(
                255 * samples.squeeze().transpose(0, 3, 1, 2),
                num_rows=num_samples // 10,
                show=True,
                save_dir=tmp_path,
                filename=f"sample_unet_diffusion_step_{steps}",
            )


def test_DiT() -> None:
    num_classes = 10
    input_shape = (1, 8, 8)
    model = DiT(
        input_shape=input_shape,
        patch_size=2,
        dim_model=8,
        num_heads=1,
        num_blocks=1,
        num_classes=num_classes,
        cfg_dropout_prob=0.1,
    )
    shape = (5, *input_shape)
    t = torch.rand((shape[0],))
    y = torch.randint(low=0, high=num_classes, size=(shape[0],))
    x = model(torch.randn(*shape), t=t, y=y)
    assert x.shape == torch.Size(shape)


def test_pos_enc_2d() -> None:
    dim_model = 16
    grid_size = 4
    shape = (10, grid_size * grid_size, dim_model)
    positional_encodings = PositionalEncoding2D(dim_model, grid_size=grid_size)
    x = positional_encodings(torch.rand(shape))
    assert x.shape == torch.Size(shape)


@pytest.mark.parametrize("classifier_free_guidance_weight", [None, 5.0])
def test_DiT_sample(classifier_free_guidance_weight) -> None:
    num_classes = 10
    class_indices = [0, 1, 3]
    num_samples = 5
    input_shape = (4, 8, 8)
    diffusion_transformer = DiT(
        input_shape=input_shape,
        patch_size=2,
        dim_model=16,
        num_heads=2,
        num_blocks=2,
        num_classes=num_classes,
        cfg_dropout_prob=0.1,
    )
    model = DiffusionModel(input_shape=input_shape, model=diffusion_transformer)
    vae = load_pretrain_vqvae()
    with suppress_plot():
        samples = generate_samples_fn_latent_cifar_10(
            model,
            vae=vae,
            class_indices=class_indices,
            num_samples=num_samples,
            mean=0.5,
            std=0.5,
            num_steps=5,
            scale_factor=1.2963932,
            cfg_weight=classifier_free_guidance_weight,
        )
    assert samples.shape == (len(class_indices), num_samples, 32, 32, 3)


def test_DiT_sample_from_loaded_model() -> None:
    num_samples = 10
    input_shape = (4, 8, 8)
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/cifar_10_latent_diffusion_DiT", filename="model.pth"
    )
    diffusion_transformer = DiT.load_model(path_to_weights).to(device)
    model = DiffusionModel(input_shape=input_shape, model=diffusion_transformer)
    vae = load_pretrain_vqvae()
    with suppress_plot():
        generate_samples_fn_latent_cifar_10(
            model,
            vae=vae,
            class_indices=list(range(10)),
            num_samples=num_samples,
            mean=0.5,
            std=0.5,
            num_steps=5,
            scale_factor=1.2963932,
            cfg_weight=None,
        )
