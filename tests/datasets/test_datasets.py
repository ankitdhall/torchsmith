import torch

from torchsmith.datahub.colored_mnist import ColoredMNISTDataset
from torchsmith.datahub.colored_mnist import ColoredMNISTWithTextDataset
from torchsmith.datahub.poetry import PoetryDataset
from torchsmith.datahub.poetry import get_huggingface_dataset
from torchsmith.tokenizers import CharacterTokenizer
from torchsmith.tokenizers.mnist_tokenizer import VQVAEMNIST
from torchsmith.tokenizers.mnist_tokenizer import ColoredMNISTImageAndTextTokenizer
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer


def test_poetry_dataset() -> None:
    sequence_length = 128
    train_df, test_df = get_huggingface_dataset()
    test_df = test_df.head(5)
    tokenizer = CharacterTokenizer.from_text(
        test_df[PoetryDataset.TEXT_COLUMN_NAME].tolist()
    )
    PoetryDataset(test_df, tokenizer, sequence_length=sequence_length)


def test_colored_mnist() -> None:
    tokenizer = VQVAEImageTokenizer(vqvae=VQVAEMNIST())
    dataset = ColoredMNISTDataset(split="test", tokenizer=tokenizer)
    assert dataset[0].shape == torch.Size([7 * 7 + 1])
    assert ColoredMNISTDataset.SEQUENCE_LENGTH == 49 + 1


def test_colored_mnist_with_text() -> None:
    tokenizer = ColoredMNISTImageAndTextTokenizer()
    dataset = ColoredMNISTWithTextDataset(split="test", tokenizer=tokenizer)
    assert dataset[0].shape == torch.Size([ColoredMNISTWithTextDataset.SEQUENCE_LENGTH])
    assert ColoredMNISTWithTextDataset.SEQUENCE_LENGTH == 60
