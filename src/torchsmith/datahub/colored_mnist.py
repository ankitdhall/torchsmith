import pickle
from itertools import chain
from typing import TYPE_CHECKING
from typing import Literal

import huggingface_hub
import numpy as np
import torch

from torchsmith.utils.pytorch import get_device

if TYPE_CHECKING:
    from torchsmith.tokenizers.mnist_tokenizer import ColoredMNISTImageAndTextTokenizer
    from torchsmith.tokenizers.mnist_tokenizer import VQVAEColoredMNISTImageTokenizer

device = get_device()


def load_colored_mnist_dataset(
    split: Literal["train", "test"],
) -> tuple[np.ndarray, list[str]]:
    assert split in ["train", "test"]
    filepath = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist",
        filename="colored_mnist_with_text.pkl",
        repo_type="dataset",
    )

    with open(filepath, "rb") as f:
        (
            train_images,
            test_images,
            train_captions,
            test_captions,
        ) = pickle.load(f)
    if split == "train":
        return train_images, train_captions
    else:
        return test_images, test_captions


def get_unique_tokens() -> set[str]:
    separator = " "
    _, text_train = load_colored_mnist_dataset("train")
    _, text_test = load_colored_mnist_dataset("test")
    text = text_train + text_test
    words = [t.split(separator) for t in text]
    tokens = set(chain.from_iterable(words))
    return tokens


class ColoredMNISTDataset(torch.utils.data.Dataset):
    SEQUENCE_LENGTH = 1 + 49

    def __init__(
        self,
        split: Literal["train", "test"],
        tokenizer: "VQVAEColoredMNISTImageTokenizer",
    ) -> None:
        images, _ = load_colored_mnist_dataset(split)  # (N, 28, 28, 3)
        self.tokenizer = tokenizer
        self.samples = torch.tensor(
            self.tokenizer.encode(images, drop_bos=False, drop_eos=True)
        )  # (N, 49 + 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class ColoredMNISTWithTextDataset(torch.utils.data.Dataset):
    IMAGE_LENGTH = 1 + 49 + 1  # 51
    TEXT_LENGTH = 1 + 6 + 1  # 8
    SEQUENCE_LENGTH = 1 + IMAGE_LENGTH + TEXT_LENGTH  # 60

    def __init__(
        self,
        split: Literal["train", "test"],
        tokenizer: "ColoredMNISTImageAndTextTokenizer",
    ) -> None:
        images, text = load_colored_mnist_dataset(split)  # (N, 28, 28, 3)
        self.tokenizer = tokenizer

        text_tokenized = self.tokenizer.tokenize_text(text)  # (N, 1 + 6 + 1)
        images_tokenized = self.tokenizer.tokenize_images(images)  # (N, 1 + 49 + 1)
        bos = torch.tensor([self.tokenizer.bos_id]).repeat((text_tokenized.shape[0], 1))

        print("self.text_tokenized.shape", text_tokenized.shape)
        print("self.images_tokenized.shape", images_tokenized.shape)

        self.text_image = torch.cat([bos, text_tokenized, images_tokenized], dim=-1)
        self.image_text = torch.cat([bos, images_tokenized, text_tokenized], dim=-1)

        self.samples = torch.cat([self.text_image, self.image_text], dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
