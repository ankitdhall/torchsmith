import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from torchsmith.datahub.arctic_monkeys import ArcticMonkeyDataset
from torchsmith.datahub.arctic_monkeys import get_huggingface_dataset
from torchsmith.datahub.text_dataset import TextDataset
from torchsmith.models.gpt2.decoder import GPT2Decoder
from torchsmith.tokenizers import CharacterTokenizer
from torchsmith.tokenizers.text_tokenizer import generate_samples_text
from torchsmith.training.config import GPT2Config
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import cross_entropy
from torchsmith.training.trainer_autoregression import TrainerAutoregression
from torchsmith.utils.dtypes import GenerateSamplesProtocol
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device

device = get_device()


def test_train_gpt2_text(tmp_path: Path) -> None:
    sequence_length = 7
    text = [
        "1 2 3",
        "4 5 6",
        "7 8 9",
    ]

    train_config = TrainConfig(num_epochs=40, batch_size=2)
    tokenizer = CharacterTokenizer.from_text(iter(text))

    transformer_config = GPT2Config(seq_len=sequence_length)
    transformer = GPT2Decoder.from_config(
        vocab_size=len(tokenizer),
        config=transformer_config,
    ).to(device)

    train_dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=transformer_config.seq_len,
    )
    data_handler = DataHandler(
        train_dataset=train_dataset,
        test_dataset=train_dataset,
        train_config=train_config,
    )

    trainer = TrainerAutoregression(
        data_handler=data_handler,
        tokenizer=tokenizer,
        train_config=train_config,
        transformer=transformer,
        loss_fn=cross_entropy,
        sequence_length=transformer_config.seq_len,
        generate_samples_fn=cast(GenerateSamplesProtocol, generate_samples_text),
        show_plots=False,
        sample_every_n_epochs=None,
        save_dir=tmp_path,
        save_every_n_epochs=5,
    )
    transformer, train_losses, test_losses, samples = trainer.train()
    print(tokenizer.decode_batch(samples.tolist()))
    plot_losses(train_losses, test_losses=test_losses, save_dir=tmp_path)


def test_train_gpt2_text_lyrics(tmp_path: Path) -> None:
    sequence_length = 32
    train_df, test_df = get_huggingface_dataset()
    tokenizer = CharacterTokenizer.from_df(
        pd.concat([train_df, test_df]), column="text"
    )
    train_config = TrainConfig(num_epochs=5, batch_size=128, num_workers=12)

    transformer_config = GPT2Config(seq_len=sequence_length)
    transformer = GPT2Decoder.from_config(
        vocab_size=len(tokenizer),
        config=transformer_config,
    ).to(device)

    train_dataset = ArcticMonkeyDataset(
        train_df, tokenizer, sequence_length=sequence_length
    )
    test_dataset = ArcticMonkeyDataset(
        test_df, tokenizer, sequence_length=sequence_length
    )
    data_handler = DataHandler(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_config=train_config,
    )

    trainer = TrainerAutoregression(
        data_handler=data_handler,
        tokenizer=tokenizer,
        train_config=train_config,
        transformer=transformer,
        loss_fn=cross_entropy,
        sequence_length=transformer_config.seq_len,
        generate_samples_fn=cast(GenerateSamplesProtocol, generate_samples_text),
        show_plots=False,
        sample_every_n_epochs=None,
        save_dir=tmp_path,
        save_every_n_epochs=5,
    )
    transformer, train_losses, test_losses, samples = trainer.train()
    print(tokenizer.decode_batch(samples.tolist()))


def test_train_gpt2_text_save_load(tmp_path: Path) -> None:
    sequence_length = 7
    text = [
        "1 2 3",
        "4 5 6",
        "7 8 9",
    ]
    transformer_config = GPT2Config(seq_len=sequence_length)
    train_config = TrainConfig(num_epochs=20, batch_size=2)
    tokenizer = CharacterTokenizer.from_text(iter(text))
    train_dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=transformer_config.seq_len,
    )
    data_handler = DataHandler(
        train_dataset=train_dataset,
        test_dataset=train_dataset,
        train_config=train_config,
    )
    transformer = GPT2Decoder.from_config(
        vocab_size=len(tokenizer),
        config=transformer_config,
    )
    trainer = TrainerAutoregression(
        data_handler=data_handler,
        tokenizer=tokenizer,
        train_config=train_config,
        transformer=transformer,
        loss_fn=cross_entropy,
        sequence_length=transformer_config.seq_len,
        generate_samples_fn=cast(GenerateSamplesProtocol, generate_samples_text),
        show_plots=False,
        sample_every_n_epochs=None,
        save_dir=tmp_path,
        save_every_n_epochs=5,
    )
    transformer, train_losses, test_losses, samples = trainer.train()
    # plot_losses(train_losses, test_losses)

    epochs_sorted = sorted(
        [
            int(p.name.split("epoch_")[-1])
            for p in trainer.save_dir.iterdir()
            if p.name.startswith("epoch_")
        ]
    )
    assert set(epochs_sorted) == {5, 10, 15, 20}
    print(f"Epochs found: {epochs_sorted}")
    for epoch_to_delete in epochs_sorted[-2:]:
        print(f"Deleted: {trainer.save_dir / f'epoch_{epoch_to_delete}'}")
        shutil.rmtree(trainer.save_dir / f"epoch_{epoch_to_delete}")

    train_config_10_epochs = TrainConfig(num_epochs=20, batch_size=2)
    trainer_10_epochs = TrainerAutoregression(
        data_handler=data_handler,
        tokenizer=tokenizer,
        train_config=train_config_10_epochs,
        transformer=transformer,
        loss_fn=cross_entropy,
        sequence_length=transformer_config.seq_len,
        generate_samples_fn=cast(GenerateSamplesProtocol, generate_samples_text),
        show_plots=False,
        sample_every_n_epochs=None,
        save_dir=tmp_path,
        save_every_n_epochs=5,
    )
    trainer_10_epochs.load_state()
    (
        transformer_10_epochs,
        train_losses_10_epochs,
        test_losses_10_epochs,
        samples_10_epochs,
    ) = trainer_10_epochs.train()

    # print(tokenizer.decode_batch(samples.tolist()))
    # plot_losses(train_losses_10_epochs, test_losses_10_epochs)
    np.testing.assert_allclose(train_losses[-1], train_losses_10_epochs[-1], atol=0.2)
    np.testing.assert_allclose(test_losses[-1], test_losses_10_epochs[-1], atol=0.2)
