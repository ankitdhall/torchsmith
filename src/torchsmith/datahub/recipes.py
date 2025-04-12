import math
import re
import unicodedata
from collections.abc import Iterator
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets import IterableDataset
from datasets import load_dataset

from torchsmith.datahub.hugging_face import HuggingFaceDataset
from torchsmith.datahub.utils import chunk_batch
from torchsmith.tokenizers import TextTokenizer
from torchsmith.utils.constants import DATA_DIR
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.pyutils import batch_function

dataset_dir = DATA_DIR / "datasets" / "recipes"


def clean_chars(text: str) -> str:
    # Regular Expression Pattern
    # - Keep: A-Z, a-z, 0-9, whitespace, basic punctuation, and square brackets
    text = text.encode("ascii", "ignore").decode("utf-8")

    # Remove ASCII control characters
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)

    # Remove non-English characters
    text = re.sub(r"[^a-zA-Z0-9\s.,?!:;'\"()\-\[\]]", "", text)
    return text


def split_to_new_line(text: str, *, keyword: str) -> str:
    try:
        return f"\n{keyword}\n".join(text.split(keyword, 1))
    except Exception:
        return text


def split_words_to_new_line(text: str, words: list[str]) -> str:
    for keyword in words:
        text = split_to_new_line(text, keyword=keyword)
    return text


def preprocess_df(df: pd.DataFrame, column_name: str, lower_case: bool) -> pd.DataFrame:
    df[column_name] = df[column_name].apply(lambda x: unicodedata.normalize("NFKD", x))
    df[column_name] = df[column_name].apply(lambda x: clean_chars(x))
    if lower_case:
        df[column_name] = df[column_name].apply(lambda x: x.lower())
    # words=["ingredients:- ", "directions:- "]
    # df[column_name] = df[column_name].apply(
    #     lambda x: split_words_to_new_line(x, words)
    # )
    # plot_length_distribution(df, column_name)
    return df


def get_huggingface_dataset(
    test_fraction: float = 0.1, lower_case: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_filename = f"train{'_lower' if lower_case else '_upper'}.csv"
    test_filename = f"test{'_lower' if lower_case else '_upper'}.csv"
    recompute = (
        not dataset_dir.exists()
        or not (dataset_dir / train_filename).exists()
        or not (dataset_dir / test_filename).exists()
    )
    if recompute:
        print(
            f"Could not find 'recipes' DataFrames at '{dataset_dir}'. Recomputing ..."
        )
        dataset_dir.mkdir(exist_ok=True, parents=True)

        # Original dataset contains only 'train' split.
        df = load_dataset("corbt/all-recipes")["train"].to_pandas()
        df = preprocess_df(df, RecipesDataset.TEXT_COLUMN_NAME, lower_case)

        # Split into 'train' and 'test' datasets.
        rng = np.random.default_rng(seed=RANDOM_STATE)
        mask_test_samples = rng.random(len(df)) <= test_fraction
        train_df, test_df = df[~mask_test_samples], df[mask_test_samples]

        # Save to disk.

        train_df.to_csv(dataset_dir / train_filename)
        test_df.to_csv(dataset_dir / test_filename)
        print(f"Saved 'recipes' DataFrames to '{dataset_dir}'.")
    else:
        print(f"Loading 'recipes' DataFrames from '{dataset_dir}' ...")
        train_df = pd.read_csv(dataset_dir / train_filename)
        test_df = pd.read_csv(dataset_dir / test_filename)

    print(f"Train dataset has {len(train_df)} rows")
    print(f"Test dataset has {len(test_df)} rows")

    return train_df, test_df


def df_to_generator(df: pd.DataFrame):
    # Convert DataFrame to a dictionary of records and yield one at a time.
    yield from df.to_dict(orient="records")


def get_iterable_dataset(num_shards: int) -> tuple[IterableDataset, IterableDataset]:
    train_df, test_df = get_huggingface_dataset()

    # Create IterableDatasets using a generator.
    train_iter_ds = Dataset.from_pandas(train_df).to_iterable_dataset(num_shards)
    test_iter_ds = Dataset.from_pandas(test_df).to_iterable_dataset(num_shards)

    return train_iter_ds, test_iter_ds


class RecipesDataset(HuggingFaceDataset):
    TEXT_COLUMN_NAME = "input"
    pass


def _get_temp_col_name(text_column_name: str) -> str:
    return text_column_name + "_temp"


def tokenize_and_chunk(
    ds: IterableDataset,
    *,
    tokenizer: TextTokenizer,
    sequence_length: int,
    verbose: bool,
    text_column_name: str = "input",
) -> IterableDataset:
    ds = ds.remove_columns(["Unnamed: 0"])

    column_name_temp = _get_temp_col_name(text_column_name)

    ds = ds.map(
        lambda samples: {
            column_name_temp: [
                tokenizer.encode(sample) for sample in samples[text_column_name]
            ]
        },
        batched=True,
        batch_size=4,
        remove_columns=[text_column_name],
    )
    ds = ds.rename_column(column_name_temp, text_column_name)

    if verbose:
        print("After tokenizing and before chunking ...")
        compute_dataset_statistics(ds, text_column_name)

    ds = ds.map(
        lambda samples: {
            column_name_temp: chunk_batch(samples[text_column_name], sequence_length)
        },
        batched=True,
        batch_size=4,
        remove_columns=[text_column_name],
    )
    ds = ds.rename_column(column_name_temp, text_column_name)

    if verbose:
        print("After tokenizing and chunking ...")
        compute_dataset_statistics(ds, text_column_name)
    return ds


def _get_num_samples_length(samples, text_column_name: str) -> tuple[int, list]:
    num_samples = 0
    sample_length = []
    for sample in samples:
        num_samples += 1
        sample_length.append(len(sample[text_column_name]))
    return num_samples, sample_length


def compute_dataset_statistics(ds: IterableDataset, text_column_name: str) -> None:
    results: Iterator[tuple[int, list]] = batch_function(
        func=partial(_get_num_samples_length, text_column_name=text_column_name),
        input=iter(ds),
        n_jobs=12,
        batch_size=20000,
    )

    num_samples = 0
    sample_length = []
    for _count, _sample_length in results:
        num_samples += _count
        sample_length.extend(_sample_length)

    # plot_histogram(sample_length, variable_name="Sample length (in tokens)")
    print(f"Number of samples: {num_samples}")
    print(f"Mean sample length: {np.mean(sample_length)} +- {np.std(sample_length)}")

    print("Showing the first item from ds:")
    print(next(iter(ds)))


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds: IterableDataset) -> None:
        super().__init__()
        self.ds = ds

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return iter(self.ds)
        else:  # in a worker process
            # split workload
            num_shards = self.ds.num_shards
            worker_id = worker_info.id
            if num_shards == worker_info.num_workers:
                return iter(self.ds.shard(self.ds.num_shards, worker_id))
            else:
                per_worker = int(math.ceil(num_shards / float(worker_info.num_workers)))
                shard_id_from = worker_id * per_worker
                shard_id_to = min(shard_id_from + per_worker, num_shards)
                datasets = [
                    self.ds.shard(self.ds.num_shards, i)
                    for i in range(shard_id_from, shard_id_to)
                ]
                return iter(chain.from_iterable(datasets))
