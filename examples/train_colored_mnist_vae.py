from functools import partial

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from torchsmith.datahub.colored_mnist import load_colored_mnist_dataset
from torchsmith.datahub.svhn import postprocess_data
from torchsmith.datahub.svhn import preprocess_data
from torchsmith.models.vae import VAEConv
from torchsmith.models.vae.utils import generate_interpolations
from torchsmith.models.vae.utils import generate_reconstructions
from torchsmith.models.vae.utils import generate_samples
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.trainer_vae_conv import VAETrainer
from torchsmith.training.utils import plot_samples
from torchsmith.utils.constants import DATA_DIR
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device

device = get_device()


class ResizedDataset(Dataset):
    def __init__(self, data, size):
        self.data = torch.tensor(data)
        self.resize = Resize(size)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.resize(x)
        return x

    def __len__(self):
        return len(self.data)


n_jobs = psutil.cpu_count()
train_config = TrainConfig(
    num_epochs=20,
    batch_size=128,
    num_workers=n_jobs,
    scheduler_config=None,
)
train_data, _ = load_colored_mnist_dataset("train")  # (N, 28, 28, 3)
test_data, _ = load_colored_mnist_dataset("test")  # (N, 28, 28, 3)

print("Before pre-processing ...")
print(
    f"train shape: {train_data.shape}, "
    f"min: {np.min(train_data)}, max: {np.max(train_data)}"
)
print(
    f"train shape: {test_data.shape}, "
    f"min: {np.min(test_data)}, max: {np.max(test_data)}"
)

train_data = (np.transpose(train_data, (0, 3, 1, 2)) / 3.0).astype("float32")
test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 3.0).astype("float32")

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

train_data = ResizedDataset(train_data, (32, 32))
test_data = ResizedDataset(test_data, (32, 32))

train_dataloader = DataLoader(
    train_data, batch_size=train_config.batch_size, shuffle=True
)
test_dataloader = DataLoader(
    test_data, batch_size=train_config.batch_size, shuffle=False
)

model = VAEConv((3, 32, 32), latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_dir = DATA_DIR / "colored_mnist_vae"
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
    sample_every_n_epochs=5,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
)
model, train_losses, test_losses, _ = trainer.train()
plot_losses(
    train_losses,
    test_losses=test_losses,
    show=True,
    labels=["-ELBO", "Reconstruction", "KL-div"],
    save_dir=experiment_dir,
)
print("Done!")

reconstructed_samples = generate_reconstructions(
    num_samples=50,
    model=model,
    dataloader=test_dataloader,
    postprocess_fn=postprocess_data,
)
plot_samples(reconstructed_samples, num_rows=10, show=True)

interpolated_samples = generate_interpolations(
    num_samples=10,
    model=model,
    dataloader=test_dataloader,
    postprocess_fn=postprocess_data,
)
plot_samples(interpolated_samples, num_rows=10, show=True)

samples = generate_samples(model, num_samples=100, postprocess_fn=postprocess_data)
