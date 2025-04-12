from functools import partial

import numpy as np
import torchvision

from torchsmith.datahub.images_with_vae import DatasetImagesWithVAE
from torchsmith.models.diffusion import DiffusionModel
from torchsmith.models.diffusion import DiT
from torchsmith.models.diffusion import generate_samples_fn_latent_cifar_10
from torchsmith.models.external.cifar_10 import load_pretrain_vqvae
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import mse
from torchsmith.training.scheduler import CosineWarmupSchedulerConfig
from torchsmith.training.trainer_diffusion import DiffusionTrainer
from torchsmith.utils.constants import DATA_DIR
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pytorch import print_named_parameters
from torchsmith.utils.pyutils import set_resource_limits

n_jobs = 12

set_resource_limits(n_jobs=n_jobs, maximum_memory=26)

device = get_device()
train_config = TrainConfig(
    num_epochs=60,
    batch_size=256,
    num_workers=n_jobs,
    scheduler_config=CosineWarmupSchedulerConfig(
        num_warmup_steps=100,
        warmup_ratio=None,
    ),
)

num_classes = 10
input_shape = (4, 8, 8)
diffusion_transformer = DiT(
    input_shape=input_shape,
    patch_size=2,
    dim_model=512,
    num_heads=8,
    num_blocks=12,
    num_classes=num_classes,
    cfg_dropout_prob=0.1,
)
model = DiffusionModel(input_shape=input_shape, model=diffusion_transformer)
vae = load_pretrain_vqvae()

print_named_parameters(diffusion_transformer)

rng = np.random.default_rng(seed=RANDOM_STATE)

train_dset = torchvision.datasets.CIFAR10(
    DATA_DIR / "cifar_dataset",
    transform=torchvision.transforms.ToTensor(),
    download=True,
    train=True,
)
test_dset = torchvision.datasets.CIFAR10(
    DATA_DIR / "cifar_dataset",
    transform=torchvision.transforms.ToTensor(),
    download=True,
    train=False,
)

train_images = train_dset.data / 255.0
train_labels = np.array(train_dset.targets, dtype=np.int32)
test_images = test_dset.data / 255.0
test_labels = np.array(test_dset.targets, dtype=np.int32)

train_data = train_images.transpose((0, 3, 1, 2))
test_data = test_images.transpose((0, 3, 1, 2))

mean, std = 0.5, 0.5
autoencoded_images = (
    vae.encode(
        (train_data[:1000] - mean) / std  # (B, C, H, W)
    )
    .cpu()
    .numpy()
)
scale_factor = float(np.std(autoencoded_images))  # 1.2963932
print(f"Using mean, std: {mean} {std} and scale factor: {scale_factor}")

train_dataset = DatasetImagesWithVAE(
    data=train_data,
    labels=train_labels,
    mean=mean,
    std=std,
    vae=vae,
    scale_factor=scale_factor,
)
test_dataset = DatasetImagesWithVAE(
    data=test_data,
    labels=test_labels,
    mean=mean,
    std=std,
    vae=vae,
    scale_factor=scale_factor,
)

experiment_dir = EXPERIMENT_DIR / "cifar_latent_diffusion"
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
    generate_samples_fn=partial(
        generate_samples_fn_latent_cifar_10,
        vae=vae,
        class_indices=list(range(10)),
        mean=mean,
        std=std,
        scale_factor=scale_factor,
    ),
    show_plots=True,
    sample_every_n_epochs=1,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
)
transformer, train_losses, test_losses, samples = trainer.train()
plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
