import unicodedata

import pandas as pd
import torch
from datasets import load_dataset

from torchsmith.datahub.utils import chunk
from torchsmith.datahub.utils import replace_punctuation
from torchsmith.tokenizers import TextTokenizer
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.pytorch import get_device

device = get_device()


def get_huggingface_dataset(
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Original dataset contains only 'train' split.
    _ds = load_dataset("huggingartists/arctic-monkeys")
    # Split into 'train' and 'test' datasets.
    ds = _ds["train"].train_test_split(test_size=test_fraction, seed=RANDOM_STATE)
    train_df, test_df = ds["train"].to_pandas(), ds["test"].to_pandas()

    for df in [train_df, test_df]:
        df["text"] = df["text"].apply(lambda x: x.lower())
        df["text"] = df["text"].apply(lambda x: unicodedata.normalize("NFKD", x))
        df["text"] = df["text"].apply(lambda x: replace_punctuation(x))

    return train_df, test_df


class ArcticMonkeyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TextTokenizer,
        sequence_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.df = df.copy()

        # Step 1: Tokenize text.
        self.df["text"] = self.df.apply(
            lambda sample: self.tokenizer.encode(sample["text"]), axis=1
        )

        # Step 2: Pre-process and drop erroneous samples.
        invalid_rows = self.df["text"].apply(lambda x: len(x) < sequence_length)
        print(
            f"Dropping {sum(invalid_rows)} rows with text shorter than "
            f"{sequence_length} tokens."
        )
        self.df = self.df[~invalid_rows]

        # Step 3: Chunk text into sequences of length `sequence_length`.
        self.df["text"] = self.df.apply(
            lambda sample: chunk(sample["text"], self.sequence_length), axis=1
        )
        self.df = self.df.explode(ignore_index=True, column="text")

        self.samples = torch.tensor(self.df["text"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
