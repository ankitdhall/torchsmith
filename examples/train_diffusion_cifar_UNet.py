import torchvision

from torchsmith.datahub.images import DatasetImages
from torchsmith.models.diffusion import DiffusionModel
from torchsmith.models.diffusion import UNet
from torchsmith.models.diffusion import generate_samples_fn_cifar_10
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import mse
from torchsmith.training.scheduler import CosineWarmupSchedulerConfig
from torchsmith.training.trainer_diffusion import DiffusionTrainer
from torchsmith.utils.constants import DATA_DIR
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import set_resource_limits

n_jobs = 12

set_resource_limits(n_jobs=n_jobs, maximum_memory=26)

device = get_device()
train_config = TrainConfig(
    num_epochs=60,
    batch_size=300,
    num_workers=n_jobs,
    scheduler_config=CosineWarmupSchedulerConfig(
        num_warmup_steps=100, warmup_ratio=None
    ),
)
unet = UNet(
    num_channels_in=3,
    num_hidden_dims=[64, 128, 256, 512],
    num_blocks_per_hidden_dim=2,
)
model = DiffusionModel(input_shape=(3, 32, 32), model=unet)


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
test_images = test_dset.data / 255.0

train_data = train_images.transpose((0, 3, 1, 2))
test_data = test_images.transpose((0, 3, 1, 2))

mean = 0.5
std = 0.5
train_dataset = DatasetImages(data=train_data, mean=mean, std=std)
test_dataset = DatasetImages(data=test_data, mean=mean, std=std)

experiment_dir = EXPERIMENT_DIR / "cifar_diffusion"
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
    generate_samples_fn=generate_samples_fn_cifar_10,
    show_plots=True,
    sample_every_n_epochs=1,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
)
transformer, train_losses, test_losses, samples = trainer.train()
plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
