import torch
from torch.distributions import Normal

from torchsmith.models.vae.vae_fc import MLP
from torchsmith.models.vae.vae_fc import VAE1D
from torchsmith.models.vae.vae_fc import negative_log_p_normal_distribution


def test_mlp() -> None:
    output_dim = 3
    mlp = MLP(2, output_shape=output_dim, hidden_dims=[10])
    input = torch.randn((5, 2))
    output = mlp(input)
    assert output.shape == torch.Size([5, output_dim])


def test_VAE1D() -> None:
    input_dim = 2
    latent_dim = 10
    hidden_dims = [16, 32]
    vae = VAE1D(
        2,
        latent_dim=latent_dim,
        encoder_hidden_dims=hidden_dims,
        decoder_hidden_dims=hidden_dims,
    )
    input = torch.randn((5, 2))
    x_reconstructed, mu_z, log_var_z, mu_x, log_var_x = vae(input)
    assert x_reconstructed.shape == torch.Size([5, input_dim])
    assert mu_z.shape == torch.Size([5, latent_dim])
    assert log_var_z.shape == torch.Size([5, latent_dim])
    assert mu_x.shape == torch.Size([5, input_dim])
    assert log_var_x.shape == torch.Size([5, input_dim])


def test_negative_log_p_normal_distribution() -> None:
    x = torch.tensor([[0.5, 1], [0.5, 0.5]])
    mu_x = torch.tensor([0.0, 0.0])
    std_x = torch.tensor([1, 1])
    value = negative_log_p_normal_distribution(x, mu_x=mu_x, log_std_x=torch.log(std_x))

    random_variable_x = Normal(mu_x, std_x)
    expected = -random_variable_x.log_prob(x)
    torch.testing.assert_close(value, expected)
