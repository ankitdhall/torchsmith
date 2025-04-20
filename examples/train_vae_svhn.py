from functools import partial

import torch
from torch.utils.data import DataLoader

from torchsmith.datahub.svhn import get_svhn
from torchsmith.datahub.svhn import postprocess_data
from torchsmith.models.vae.utils import generate_interpolations
from torchsmith.models.vae.utils import generate_reconstructions
from torchsmith.models.vae.utils import generate_samples
from torchsmith.models.vae.vae_conv import VAEConv
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.trainer_vae_conv import VAETrainer
from torchsmith.training.utils import plot_samples
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import set_resource_limits

n_jobs = 12
set_resource_limits(n_jobs=n_jobs, maximum_memory=26)
device = get_device()


train_config = TrainConfig(
    num_epochs=10,
    batch_size=128,
    num_workers=n_jobs,
    scheduler_config=None,
)

train_data = get_svhn(split="train")
test_data = get_svhn(split="test")

train_dataloader = DataLoader(
    train_data, batch_size=train_config.batch_size, shuffle=True
)
test_dataloader = DataLoader(
    test_data, batch_size=train_config.batch_size, shuffle=False
)

model = VAEConv((3, 32, 32), latent_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_dir = EXPERIMENT_DIR / "svhn_vae"
print(f"Saving experiment to: {experiment_dir}")

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
    sample_every_n_epochs=10000,
    save_dir=experiment_dir,
    save_every_n_epochs=10000,
)
model, train_losses, test_losses, _ = trainer.train()
print("Done!")


reconstructed_samples = generate_reconstructions(
    num_samples=50,
    model=model,
    dataloader=test_dataloader,
    postprocess_fn=postprocess_data,
)
plot_samples(reconstructed_samples, num_rows=10, show=True)

interpolated_samples = generate_interpolations(
    model=model,
    num_samples=10,
    dataloader=test_dataloader,
    postprocess_fn=postprocess_data,
)
plot_samples(interpolated_samples, num_rows=10, show=True)

samples = generate_samples(model, num_samples=100, postprocess_fn=postprocess_data)

plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
