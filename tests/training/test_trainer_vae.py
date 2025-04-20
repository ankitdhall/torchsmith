from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader

from torchsmith.datahub.svhn import postprocess_data
from torchsmith.datahub.svhn import preprocess_data
from torchsmith.models.vae.utils import generate_samples
from torchsmith.models.vae.vae_conv import VAEConv
from torchsmith.models.vae.vae_fc import MLP
from torchsmith.models.vae.vae_fc import VAE1D
from torchsmith.models.vae.vae_fc import negative_log_p_normal_distribution
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.trainer_vae_conv import VAETrainer
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


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
    mu_z, log_var_z, mu_x, log_var_x = vae(input)
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


def test_train_vae(tmp_path: Path) -> None:
    train_config = TrainConfig(
        num_epochs=3,
        batch_size=128,
        num_workers=4,
        scheduler_config=None,
    )
    rng = np.random.default_rng(seed=RANDOM_STATE)
    train_data = preprocess_data(rng.random((20, 3, 32, 32)).astype("float32"))
    test_data = preprocess_data(rng.random((20, 3, 32, 32)).astype("float32"))

    train_dataloader = DataLoader(
        train_data, batch_size=train_config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=train_config.batch_size, shuffle=False
    )

    model = VAEConv((3, 32, 32), latent_dim=16).to(device)

    experiment_dir = tmp_path
    data_handler = DataHandler(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_config=train_config,
    )

    trainer = VAETrainer(
        data_handler=data_handler,
        train_config=train_config,
        model=model,
        generate_samples_fn=partial(generate_samples, postprocess_fn=postprocess_data),
        sample_every_n_epochs=1,
        save_dir=experiment_dir,
        save_every_n_epochs=1,
    )
    with suppress_plot():
        model, train_losses, test_losses, _ = trainer.train()
        plot_losses(
            train_losses,
            test_losses=test_losses,
            show=True,
            labels=["-ELBO", "Reconstruction", "KL-div"],
            save_dir=experiment_dir,
        )
