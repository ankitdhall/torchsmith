import huggingface_hub
import numpy as np
import pytest
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader

from torchsmith.datahub.svhn import postprocess_data
from torchsmith.datahub.svhn import preprocess_data
from torchsmith.models.vae.utils import generate_interpolations
from torchsmith.models.vae.utils import generate_reconstructions
from torchsmith.models.vae.utils import generate_samples
from torchsmith.models.vae.vae_conv import VAEConv
from torchsmith.models.vae.vae_fc import MLP
from torchsmith.models.vae.vae_fc import VAE1D
from torchsmith.models.vae.vae_fc import negative_log_p_normal_distribution
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


def test_VAE1D() -> None:
    input_dim = 2
    latent_dim = 10
    hidden_dims = [16, 32]
    vae = VAE1D(
        input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=hidden_dims,
        decoder_hidden_dims=hidden_dims,
    )
    input = torch.randn((5, 2))
    mu_z, log_var_z, mu_x, log_var_x = vae(input)
    assert mu_z.shape == torch.Size([5, latent_dim])
    assert log_var_z.shape == torch.Size([5, latent_dim])
    assert mu_x.shape == torch.Size([5, input_dim])
    assert log_var_x.shape == torch.Size([5, input_dim])

    loss, loss_reconstruction, loss_kl_div = vae.loss(input)
    assert loss == loss_reconstruction + loss_kl_div


@pytest.mark.parametrize("add_noise", [True, False])
def test_VAE1D_sampling(add_noise: bool) -> None:
    input_dim = 2
    latent_dim = 10
    hidden_dims = [16, 32]
    vae = VAE1D(
        input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=hidden_dims,
        decoder_hidden_dims=hidden_dims,
    )
    num_samples = 10
    samples = vae.sample(num_samples=num_samples, add_noise=add_noise)
    assert samples.shape == torch.Size([num_samples, input_dim])


def test_negative_log_p_normal_distribution() -> None:
    x = torch.tensor([[0.5, 1], [0.5, 0.5]])
    mu_x = torch.tensor([0.0, 0.0])
    std_x = torch.tensor([1, 1])
    value = negative_log_p_normal_distribution(x, mu_x=mu_x, log_std_x=torch.log(std_x))

    random_variable_x = Normal(mu_x, std_x)
    expected = -random_variable_x.log_prob(x)
    torch.testing.assert_close(value, expected)


def test_mlp() -> None:
    output_dim = 3
    mlp = MLP(2, output_shape=output_dim, hidden_dims=[10])
    input = torch.randn((5, 2))
    output = mlp(input)
    assert output.shape == torch.Size([5, output_dim])


def test_VAEConv() -> None:
    num_samples = 10
    input_shape = (3, 32, 32)
    latent_dim = 16
    vae = VAEConv(input_shape, latent_dim=latent_dim).to(device)
    input = (torch.rand((num_samples, *input_shape)) - 0.5) * 2  # [0, 1] -> [-1, 1]
    mu_z, log_std_z, x, x_reconstructed = vae(input)
    assert x_reconstructed.shape == torch.Size([num_samples, *input_shape])
    torch.testing.assert_close(x, input)
    assert mu_z.shape == torch.Size([num_samples, latent_dim])
    assert log_std_z.shape == torch.Size([num_samples, latent_dim])


def test_generate_samples() -> None:
    num_samples = 16
    input_shape = (3, 32, 32)
    latent_dim = 16
    vae = VAEConv(input_shape, latent_dim=latent_dim).to(device)
    with suppress_plot():
        samples = generate_samples(
            vae, num_samples=num_samples, postprocess_fn=postprocess_data
        )
    assert samples.shape == torch.Size([num_samples, *input_shape])


def test_generate_reconstructions() -> None:
    num_samples = 5
    input_shape = (3, 32, 32)
    latent_dim = 16
    vae = VAEConv(input_shape, latent_dim=latent_dim).to(device)

    rng = np.random.default_rng(seed=RANDOM_STATE)
    data = preprocess_data(rng.random((20, 3, 32, 32)).astype("float32"))
    dataloader = DataLoader(data, batch_size=10, shuffle=True)
    with suppress_plot():
        samples = generate_reconstructions(
            num_samples=num_samples,
            model=vae,
            dataloader=dataloader,
            postprocess_fn=postprocess_data,
        )
    assert samples.shape == torch.Size([2 * num_samples, *input_shape])


def test_generate_interpolations() -> None:
    num_samples = 5
    num_steps = 5
    input_shape = (3, 32, 32)
    latent_dim = 16
    vae = VAEConv(input_shape, latent_dim=latent_dim).to(device)

    rng = np.random.default_rng(seed=RANDOM_STATE)
    data = preprocess_data(rng.random((20, 3, 32, 32)).astype("float32"))
    dataloader = DataLoader(data, batch_size=10, shuffle=True)
    with suppress_plot():
        samples = generate_interpolations(
            num_samples=num_samples,
            model=vae,
            dataloader=dataloader,
            postprocess_fn=postprocess_data,
            num_steps=num_steps,
        )
    assert samples.shape == torch.Size([num_steps * num_samples, *input_shape])


def test_VAEConv_sample_from_loaded_model() -> None:
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/svhn_vae", filename="model.pth"
    )
    vae = VAEConv.load_model(path_to_weights).to(device)

    num_samples = 10
    input_shape = (3, 32, 32)
    with suppress_plot():
        samples = generate_samples(
            vae, num_samples=num_samples, postprocess_fn=postprocess_data
        )
    assert samples.shape == torch.Size([num_samples, *input_shape])
