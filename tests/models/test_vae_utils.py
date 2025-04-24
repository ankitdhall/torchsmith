import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from torchsmith.datahub.svhn import postprocess_data
from torchsmith.datahub.svhn import preprocess_data
from torchsmith.models.vae.base import BaseVAE
from torchsmith.models.vae.utils import generate_reconstructions
from torchsmith.models.vae.utils import generate_samples
from torchsmith.models.vae.vae_conv import VAEConv
from torchsmith.models.vae.vqvae import VQVAE
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


@pytest.mark.parametrize(
    ["model", "input_shape"],
    [
        (VAEConv((3, 32, 32), latent_dim=16).to(device), (3, 32, 32)),
        (VQVAE((3, 32, 32), latent_dim=16, codebook_size=10).to(device), (3, 32, 32)),
    ],
)
def test_generate_samples(model: BaseVAE, input_shape: tuple[int, ...]) -> None:
    num_samples = 16
    with suppress_plot():
        samples = generate_samples(
            model, num_samples=num_samples, postprocess_fn=postprocess_data
        )
    assert samples.shape == torch.Size([num_samples, *input_shape])


@pytest.mark.parametrize(
    ["model", "input_shape"],
    [
        (VAEConv((3, 32, 32), latent_dim=16).to(device), (3, 32, 32)),
        (VQVAE((3, 32, 32), latent_dim=16, codebook_size=10).to(device), (3, 32, 32)),
    ],
)
def test_generate_reconstructions(model: BaseVAE, input_shape: tuple[int, ...]) -> None:
    num_samples = 5
    rng = np.random.default_rng(seed=RANDOM_STATE)
    data = preprocess_data(rng.random((20, 3, 32, 32)).astype("float32"))
    dataloader = DataLoader(data, batch_size=10, shuffle=True)
    with suppress_plot():
        samples = generate_reconstructions(
            num_samples=num_samples,
            model=model,
            dataloader=dataloader,
            postprocess_fn=postprocess_data,
        )
    assert samples.shape == torch.Size([2 * num_samples, *input_shape])
