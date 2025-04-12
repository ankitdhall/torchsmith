import pandas as pd
from datasets import load_dataset

from torchsmith.datahub.hugging_face import HuggingFaceDataset
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.pytorch import get_device

device = get_device()


def get_huggingface_dataset(
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Original dataset contains only 'train' split.
    _ds = load_dataset("merve/poetry")
    # Split into 'train' and 'test' datasets.
    ds = _ds["train"].train_test_split(test_size=test_fraction, seed=RANDOM_STATE)
    train_df, test_df = ds["train"].to_pandas(), ds["test"].to_pandas()

    return train_df, test_df


class PoetryDataset(HuggingFaceDataset):
    TEXT_COLUMN_NAME = "content"
