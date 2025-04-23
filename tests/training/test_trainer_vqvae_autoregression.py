from functools import partial
from pathlib import Path

import numpy as np

from torchsmith.datahub.images_with_vqvae import ImagesWithVQVAEDataset
from torchsmith.datahub.svhn import preprocess_data
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.models.vae.vq_vae import VQVAE
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.tokenizers.vqvae_tokenizer import generate_samples_image
from torchsmith.training.config import GPT2Config
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import cross_entropy
from torchsmith.training.trainer_autoregression import TrainerAutoregression
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device

device = get_device()


def postprocess_data_transformer(x: np.ndarray) -> np.ndarray:
    # Assume x in [-1, 1]
    x = np.clip(x, a_min=-1, a_max=1)
    x = (x / 2) + 0.5  # -> [0, 1]
    x = np.transpose(x, (0, 2, 3, 1))
    return x  # in [0, 1]


def test_vqvae_with_autoregression(tmp_path: Path) -> None:
    n_jobs = 4
    rng = np.random.default_rng(seed=RANDOM_STATE)
    train_data = preprocess_data(rng.random((20, 3, 32, 32)).astype("float32"))
    test_data = preprocess_data(rng.random((20, 3, 32, 32)).astype("float32"))

    vqvae = VQVAE((3, 32, 32), latent_dim=256, codebook_size=128).to(device)
    vqvae_tokenizer = VQVAEImageTokenizer(vqvae=vqvae, batch_size=10000)

    train_data_tokens = ImagesWithVQVAEDataset(train_data, tokenizer=vqvae_tokenizer)
    test_data_tokens = ImagesWithVQVAEDataset(test_data, tokenizer=vqvae_tokenizer)

    print(f"train_data_tokens.sequence_length: {train_data_tokens.sequence_length}")

    train_config_transformer = TrainConfig(
        num_epochs=5, batch_size=512, num_workers=n_jobs
    )
    transformer_config = GPT2Config(seq_len=train_data_tokens.sequence_length)
    transformer = GPT2Decoder.from_config(
        vocab_size=len(vqvae_tokenizer),
        config=transformer_config,
    )

    experiment_dir = tmp_path / "svhn_vqvae_transformer"
    print(f"Saving experiment to: {experiment_dir}")

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
            generate_samples_image,
            postprocess_fn=postprocess_data_transformer,
            num_samples=25,
        ),
        show_plots=False,
        sample_every_n_epochs=1,
        save_dir=experiment_dir,
        save_every_n_epochs=1,
    )
    (
        transformer,
        train_losses_transformer,
        test_losses_transformer,
        samples_transformer,
    ) = trainer_transformer.train()
    plot_losses(
        train_losses_transformer,
        test_losses=test_losses_transformer,
        save_dir=experiment_dir,
        show=True,
    )
