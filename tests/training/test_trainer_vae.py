import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from torchsmith.datahub.svhn import postprocess_data
from torchsmith.datahub.svhn import preprocess_data
from torchsmith.models.vae.base import BaseVAE
from torchsmith.models.vae.utils import generate_samples
from torchsmith.models.vae.vae_conv import VAEConv
from torchsmith.models.vae.vqvae import VQVAE
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.trainer_vae_conv import VAETrainer
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


@pytest.mark.parametrize(
    "model",
    [
        VAEConv((3, 32, 32), latent_dim=16).to(device),
        VQVAE((3, 32, 32), latent_dim=16, codebook_size=10).to(device),
    ],
)
def test_train_vae(model: BaseVAE, tmp_path: Path) -> None:
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

    epochs_sorted = sorted(
        [
            int(p.name.split("epoch_")[-1])
            for p in trainer.save_dir.iterdir()
            if p.name.startswith("epoch_")
        ]
    )
    assert set(epochs_sorted) == set(list(range(1, 4)))
    print(f"Epochs found: {epochs_sorted}")
    for epoch_to_delete in epochs_sorted[-2:]:
        print(f"Deleted: {trainer.save_dir / f'epoch_{epoch_to_delete}'}")
        shutil.rmtree(trainer.save_dir / f"epoch_{epoch_to_delete}")

    trainer.load_state()
    with suppress_plot():
        model, train_losses, test_losses, _ = trainer.train()
        plot_losses(
            train_losses,
            test_losses=test_losses,
            show=True,
            labels=["-ELBO", "Reconstruction", "KL-div"],
            save_dir=experiment_dir,
        )
