from functools import partial

import numpy as np

from torchsmith.models.diffusion import MLP
from torchsmith.models.diffusion import Dataset2D
from torchsmith.models.diffusion import DiffusionModel
from torchsmith.models.diffusion import generate_samples_fn_2d
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import mse
from torchsmith.training.scheduler import CosineWarmupSchedulerConfig
from torchsmith.training.trainer_diffusion import DiffusionTrainer
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import set_resource_limits

n_jobs = 12
set_resource_limits(n_jobs=n_jobs, maximum_memory=26)

device = get_device()
train_config = TrainConfig(
    num_epochs=10,
    batch_size=1024,
    num_workers=n_jobs,
    scheduler_config=CosineWarmupSchedulerConfig(num_warmup_steps=5, warmup_ratio=None),
)
model = DiffusionModel(input_shape=2, model=MLP(input_dim=2 + 1, output_dim=2))

rng = np.random.default_rng(seed=RANDOM_STATE)
train_data = rng.normal(size=(1000, 2))
test_data = rng.normal(size=(1000, 2))
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_dataset = Dataset2D(data=train_data, mean=mean, std=std)
test_dataset = Dataset2D(data=test_data, mean=mean, std=std)

experiment_dir = EXPERIMENT_DIR / "2d_diffusion"
print(f"Saving experiment to: {experiment_dir}")

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
    sample_every_n_epochs=1,
    save_dir=experiment_dir,
    save_every_n_epochs=2,
)
transformer, train_losses, test_losses, samples = trainer.train()
print(samples)
plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
