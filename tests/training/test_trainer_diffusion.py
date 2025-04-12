import shutil
from functools import partial
from pathlib import Path

import numpy as np

from torchsmith.models.diffusion import MLP
from torchsmith.models.diffusion import Dataset2D
from torchsmith.models.diffusion import generate_samples_fn_2d
from torchsmith.models.diffusion.diffusion import DiffusionModel
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import mse
from torchsmith.training.scheduler import CosineWarmupSchedulerConfig
from torchsmith.training.trainer_diffusion import DiffusionTrainer
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


def test_train_diffusion_model_save_load(tmp_path: Path) -> None:
    model = DiffusionModel(input_shape=2, model=MLP(input_dim=2 + 1, output_dim=2))

    rng = np.random.default_rng(seed=RANDOM_STATE)
    train_data = rng.normal(size=(1000, 2))
    test_data = rng.normal(size=(1000, 2))
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_dataset = Dataset2D(data=train_data, mean=mean, std=std)
    test_dataset = Dataset2D(data=test_data, mean=mean, std=std)

    experiment_dir = tmp_path / "2d_diffusion"
    print(f"Saving experiment to: {experiment_dir}")

    train_config = TrainConfig(
        num_epochs=5,
        batch_size=1024,
        num_workers=4,
        scheduler_config=CosineWarmupSchedulerConfig(
            num_warmup_steps=2, warmup_ratio=None
        ),
    )

    data_handler = DataHandler(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_config=train_config,
    )
    trainer = DiffusionTrainer(
        data_handler=data_handler,
        train_config=train_config,
        model=model,
        loss_fn=mse,
        generate_samples_fn=partial(generate_samples_fn_2d, mean=mean, std=std),
        show_plots=False,
        sample_every_n_epochs=None,
        save_dir=experiment_dir,
        save_every_n_epochs=1,
    )
    model_trained, train_losses, test_losses, samples = trainer.train()
    with suppress_plot():
        plot_losses(
            train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True
        )

    epochs_sorted = sorted(
        [
            int(p.name.split("epoch_")[-1])
            for p in trainer.save_dir.iterdir()
            if p.name.startswith("epoch_")
        ]
    )
    assert set(epochs_sorted) == set(list(range(1, 6)))
    print(f"Epochs found: {epochs_sorted}")
    for epoch_to_delete in epochs_sorted[-2:]:
        print(f"Deleted: {trainer.save_dir / f'epoch_{epoch_to_delete}'}")
        shutil.rmtree(trainer.save_dir / f"epoch_{epoch_to_delete}")

    trainer = DiffusionTrainer(
        data_handler=data_handler,
        train_config=train_config,
        model=model,
        loss_fn=mse,
        generate_samples_fn=partial(generate_samples_fn_2d, mean=mean, std=std),
        show_plots=False,
        sample_every_n_epochs=None,
        save_dir=experiment_dir,
        save_every_n_epochs=1,
    )
    trainer.load_state()
    (
        model_trained_10_epochs,
        train_losses_10_epochs,
        test_losses_10_epochs,
        samples_10_epochs,
    ) = trainer.train()

    np.testing.assert_allclose(train_losses[-1], train_losses_10_epochs[-1], atol=0.2)
    np.testing.assert_allclose(test_losses[-1], test_losses_10_epochs[-1], atol=0.2)
