import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from torchsmith.models.vae.vae_conv import VAEConv
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.trainer_vae_conv import VAETrainer
from torchsmith.training.utils import plot_samples
from torchsmith.utils.constants import DATA_DIR
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import set_resource_limits

n_jobs = 12

set_resource_limits(n_jobs=n_jobs, maximum_memory=26)

device = get_device()


train_dataset = torchvision.datasets.SVHN(
    root=DATA_DIR / "svhn",
    split="train",
    download=True,
    transform=transforms.ToTensor(),
)
test_dataset = torchvision.datasets.SVHN(
    root=DATA_DIR / "svhn", split="test", download=True, transform=transforms.ToTensor()
)
train_data = train_dataset.data.transpose((0, 2, 3, 1))
test_data = test_dataset.data.transpose((0, 2, 3, 1))


def preprocess_data(x: np.ndarray) -> np.ndarray:
    # Assume x in [0, 1]
    x = x - 0.5  # [0, 1] -> [-0.5, 0.5]
    x = x * 2  # [-0.5, 0.5] -> [-1, 1]
    return x  # in [-1, 1]


def postprocess_data(x: np.ndarray) -> np.ndarray:
    # Assume x in [-1, 1]
    x = np.clip(x, a_min=-1, a_max=1)
    x = (x / 2) + 0.5  # -> [0, 1]
    x = (x * 255).astype(int)
    x = np.transpose(x, (0, 2, 3, 1))
    return x  # in [0, 255]


def generate_samples(model: VAEConv, *, num_samples: int = 100) -> np.ndarray:
    samples = postprocess_data(model.sample(num_samples))
    plot_samples(samples, num_rows=int(num_samples**0.5), show=True)
    return samples


def generate_reconstructions(
    num_samples: int, model: VAEConv, dataloader: torch.utils.data.DataLoader
) -> np.ndarray:
    x = next(iter(dataloader))[:num_samples].to(device)
    with torch.no_grad():
        z, _ = model.encoder(x)
        x_recon = model.decoder(z)
    reconstructions = np.stack((x.cpu(), x_recon.cpu()), axis=1).reshape(
        (-1, 3, 32, 32)
    )
    reconstructions = postprocess_data(reconstructions)
    return reconstructions


def generate_interpolations(
    num_samples: int,
    model: VAEConv,
    dataloader: torch.utils.data.DataLoader,
    num_steps: int = 10,
) -> np.ndarray:
    x = next(iter(dataloader))[: 2 * num_samples].to(device)
    with torch.no_grad():
        z, _ = model.encoder(x)  # (20, N_l)
        z1, z2 = z.chunk(2, dim=0)  # (20, N_l) -> # (10, N_l), (10, N_l)
        interpolations = [
            model.decoder(z1 * (1 - alpha) + z2 * alpha).cpu()
            for alpha in np.linspace(0, 1, num_steps)
        ]
        interpolations = np.stack(interpolations, axis=1).reshape((-1, 3, 32, 32))
    interpolations = postprocess_data(interpolations)
    return interpolations


train_config = TrainConfig(
    num_epochs=10,
    batch_size=128,
    num_workers=n_jobs,
    scheduler_config=None,
)

print("Before pre-processing ...")
print(
    f"train shape: {train_data.shape}, "
    f"min: {np.min(train_data)}, max: {np.max(train_data)}"
)
print(
    f"train shape: {test_data.shape}, "
    f"min: {np.min(test_data)}, max: {np.max(test_data)}"
)

train_data = (np.transpose(train_data, (0, 3, 1, 2)) / 255.0).astype("float32")
test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255.0).astype("float32")

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

print("After pre-processing ...")
print(
    f"train shape: {train_data.shape}, "
    f"min: {np.min(train_data)}, max: {np.max(train_data)}"
)
print(
    f"train shape: {test_data.shape}, "
    f"min: {np.min(test_data)}, max: {np.max(test_data)}"
)

train_dataloader = DataLoader(
    train_data, batch_size=train_config.batch_size, shuffle=True
)
test_dataloader = DataLoader(
    test_data, batch_size=train_config.batch_size, shuffle=False
)

model = VAEConv((3, 32, 32), latent_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_dir = EXPERIMENT_DIR / "vae_svhn_diffusion"
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
    generate_samples_fn=generate_samples,
    sample_every_n_epochs=10000,
    save_dir=experiment_dir,
    save_every_n_epochs=1000,
)
model, train_losses, test_losses, _ = trainer.train()
print("Done!")


reconstructed_samples = generate_reconstructions(
    num_samples=50, model=model, dataloader=test_dataloader
)
plot_samples(reconstructed_samples, num_rows=10, show=True)

interpolated_samples = generate_interpolations(
    num_samples=10, model=model, dataloader=test_dataloader
)
plot_samples(interpolated_samples, num_rows=10, show=True)

samples = generate_samples(model, num_samples=100)

plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
