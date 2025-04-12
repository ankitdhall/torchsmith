import re
import unicodedata

import pandas as pd
from datasets import load_dataset

from torchsmith.datahub.hugging_face import HuggingFaceDataset
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.pytorch import get_device

device = get_device()


def clean_chars(text: str) -> str:
    # Regular Expression Pattern
    # - Keep: A-Z, a-z, 0-9, whitespace, basic punctuation, and square brackets
    text = text.encode("ascii", "ignore").decode("utf-8")
    # Remove non-English characters
    text = re.sub(r"[^a-zA-Z0-9\s.,?!:;'\"()\-\[\]]", "", text)
    return text


def get_huggingface_dataset(
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Original dataset contains only 'train' split.
    _ds = load_dataset("huggingartists/eminem")
    # Split into 'train' and 'test' datasets.
    ds = _ds["train"].train_test_split(test_size=test_fraction, seed=RANDOM_STATE)
    train_df, test_df = ds["train"].to_pandas(), ds["test"].to_pandas()
    column_name = EminemDataset.TEXT_COLUMN_NAME

    def _remove_prefix(text: str) -> str:
        # All lyrics have a prefix "{SONG_NAME} Lyrics " that we want to remove.
        return text.split(" Lyrics\n", 1)[-1]

    def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        df[column_name] = df[column_name].apply(lambda x: _remove_prefix(x))
        df[column_name] = df[column_name].apply(
            lambda x: unicodedata.normalize("NFKD", x)
        )
        df[column_name] = df[column_name].apply(lambda x: clean_chars(x))
        # plot_length_distribution(df, column_name)
        min_num_chars_per_sample = 200
        max_num_chars_per_sample = 7000
        invalid_samples_mask = (
            df[column_name].apply(lambda x: len(x)) < min_num_chars_per_sample
        )
        print(
            f"Removing {invalid_samples_mask.sum()} "
            f"({invalid_samples_mask.mean() * 100:.3f}%) samples with less than "
            f"{min_num_chars_per_sample} characters."
        )
        df = df[~invalid_samples_mask]
        invalid_samples_mask = (
            df[column_name].apply(lambda x: len(x)) > max_num_chars_per_sample
        )
        print(
            f"Removing {invalid_samples_mask.sum()} "
            f"({invalid_samples_mask.mean() * 100:.3f}%) samples with more than "
            f"{max_num_chars_per_sample} characters."
        )
        df = df[~invalid_samples_mask]
        return df

    train_df = preprocess_df(train_df)
    test_df = preprocess_df(test_df)

    return train_df, test_df


class EminemDataset(HuggingFaceDataset):
    pass
