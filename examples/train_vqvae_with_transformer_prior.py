from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchsmith.datahub.images_with_vqvae import ImagesWithVQVAEDataset
from torchsmith.datahub.svhn import get_svhn
from torchsmith.datahub.svhn import postprocess_data
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.models.vae import VQVAE
from torchsmith.models.vae import BaseVAE
from torchsmith.models.vae.utils import generate_reconstructions
from torchsmith.models.vae.utils import generate_samples
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.tokenizers.vqvae_tokenizer import generate_samples_image_v2
from torchsmith.training.config import GPT2Config
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import cross_entropy
from torchsmith.training.scheduler import CosineWarmupSchedulerConfig
from torchsmith.training.trainer_autoregression import TrainerAutoregression
from torchsmith.training.trainer_vae_conv import VAETrainer
from torchsmith.training.utils import plot_samples
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import set_resource_limits

n_jobs = 12
set_resource_limits(n_jobs=n_jobs, maximum_memory=26)
device = get_device()


def generate_reconstructions_wrapped(
    model: BaseVAE, dataloader: torch.utils.data.DataLoader
) -> None:
    reconstructions = generate_reconstructions(
        model=model,
        num_samples=50,
        dataloader=dataloader,
        postprocess_fn=postprocess_data,
    )
    plot_samples(reconstructions, num_rows=10, show=True)


dataset_name = "svhn"
train_data = get_svhn(split="train")
test_data = get_svhn(split="test")

print("Loaded dataset (incl. pre-processing) ...")
print(
    f"train shape: {train_data.shape}, "
    f"min: {np.min(train_data)}, max: {np.max(train_data)}"
)
print(
    f"train shape: {test_data.shape}, "
    f"min: {np.min(test_data)}, max: {np.max(test_data)}"
)

train_config = TrainConfig(
    num_epochs=20,
    batch_size=128,
    num_workers=n_jobs,
    scheduler_config=None,
)
train_dataloader = DataLoader(
    train_data, batch_size=train_config.batch_size, shuffle=True
)
test_dataloader = DataLoader(
    test_data, batch_size=train_config.batch_size, shuffle=False
)


experiment_dir = EXPERIMENT_DIR / f"vqvae_{dataset_name}"
print(f"Saving experiment to: {experiment_dir}")

vqvae = VQVAE((3, 32, 32), latent_dim=256, codebook_size=128).to(device)
data_handler = DataHandler(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    train_config=train_config,
)
trainer_vqvae = VAETrainer(
    data_handler=data_handler,
    train_config=train_config,
    model=vqvae,
    generate_samples_fn=partial(
        generate_reconstructions_wrapped, dataloader=test_dataloader
    ),
    sample_every_n_epochs=1,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
)

vqvae, train_losses, test_losses, _ = trainer_vqvae.train()
plot_losses(
    train_losses,
    test_losses=test_losses,
    show=True,
    labels=["Total loss", "Reconstruction", "Encoder<->Codebook"],
    save_dir=experiment_dir,
)
print("Finished training the VQVAE")

reconstructed_samples = generate_reconstructions(
    num_samples=50,
    model=vqvae,
    dataloader=test_dataloader,
    postprocess_fn=postprocess_data,
)
plot_samples(reconstructed_samples, num_rows=10, show=True)

samples = generate_samples(vqvae, num_samples=100, postprocess_fn=postprocess_data)
plot_samples(samples, num_rows=10, show=True)


print(
    """
    ####################################################
    ############### Transformer training ###############
    ####################################################
    """
)

# Instantiate the tokenizer using the trained VQVAE model
vqvae_tokenizer = VQVAEImageTokenizer(vqvae=vqvae, batch_size=1000)

train_data_tokens = ImagesWithVQVAEDataset(train_data, tokenizer=vqvae_tokenizer)
test_data_tokens = ImagesWithVQVAEDataset(test_data, tokenizer=vqvae_tokenizer)

print(f"train_data_tokens: {train_data_tokens.samples}")
print(f"test_data_tokens: {test_data_tokens.samples}")
print(f"train_data_tokens.sequence_length: {train_data_tokens.sequence_length}")

experiment_dir = EXPERIMENT_DIR / f"transformer_prior_{dataset_name}"
print(f"Saving experiment to: {experiment_dir}")

train_config_transformer = TrainConfig(
    num_epochs=20,
    batch_size=128,
    num_workers=n_jobs,
    scheduler_config=CosineWarmupSchedulerConfig(
        num_warmup_steps=1000, warmup_ratio=None
    ),
)
transformer_config = GPT2Config(seq_len=train_data_tokens.sequence_length)
transformer = GPT2Decoder.from_config(
    vocab_size=len(vqvae_tokenizer),
    config=transformer_config,
)
data_handler_transformer = DataHandler(
    train_dataset=train_data_tokens,
    test_dataset=test_data_tokens,
    train_config=train_config_transformer,
)
trainer_transformer = TrainerAutoregression(
    data_handler=data_handler_transformer,
    tokenizer=vqvae_tokenizer,
    train_config=train_config_transformer,
    transformer=transformer,
    loss_fn=cross_entropy,
    sequence_length=transformer_config.seq_len,
    generate_samples_fn=partial(
        generate_samples_image_v2,
        postprocess_fn=postprocess_data,
        num_samples=100,
    ),
    show_plots=False,
    sample_every_n_epochs=1,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
)
transformer, train_losses_transformer, test_losses_transformer, samples_transformer = (
    trainer_transformer.train()
)
plot_losses(
    train_losses_transformer,
    test_losses=test_losses_transformer,
    save_dir=experiment_dir,
    show=True,
)

plot_samples(samples_transformer.cpu().numpy())
print("Finished training the transformer prior")
