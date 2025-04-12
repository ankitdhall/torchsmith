from itertools import chain

import pandas as pd
import torch

from torchsmith.datahub.utils import chunk
from torchsmith.tokenizers import TextTokenizer
from torchsmith.utils.pytorch import get_device

device = get_device()


class HuggingFaceDataset(torch.utils.data.Dataset):
    TEXT_COLUMN_NAME = "text"

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TextTokenizer,
        sequence_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.df = df.copy()

        self.df[self.TEXT_COLUMN_NAME] = self.df.apply(
            lambda sample: self.tokenizer.encode(sample[self.TEXT_COLUMN_NAME]), axis=1
        )

        flattened_list = list(
            chain.from_iterable(self.df[self.TEXT_COLUMN_NAME].tolist())
        )
        chunked_samples = chunk(flattened_list, self.sequence_length)
        self.samples = torch.tensor(chunked_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
