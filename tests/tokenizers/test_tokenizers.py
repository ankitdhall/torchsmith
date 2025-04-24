from collections import Counter
from itertools import chain
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from torchsmith.tokenizers.byte_pair_encoding_tokenizer import BytePairEncodingBuilder
from torchsmith.tokenizers.byte_pair_encoding_tokenizer import replace_pair
from torchsmith.tokenizers.character_tokenizer import CharacterTokenizer
from torchsmith.tokenizers.mnist_tokenizer import VQVAEMNIST
from torchsmith.tokenizers.string_tokenizer import StringTokenizer
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.utils.constants import RANDOM_STATE


@pytest.mark.parametrize(["n_jobs", "batch_size"], [(1, 2), (2, 2)])
def test_character_tokenizer(n_jobs: int, batch_size: int) -> None:
    text = [
        "1 2 3",
        "4 5 6",
        "7 8 9",
    ]
    tokenizer = CharacterTokenizer.from_text(
        iter(text), n_jobs=n_jobs, batch_size=batch_size
    )
    assert len(tokenizer) == 9 + 1 + 1 + 1  # 9 digits + 1 bos + 1 eos + 1 space
    text_encoded = list(tokenizer.encode_batch(iter(text)))
    assert set(np.unique(text_encoded)).issubset(set(range(len(tokenizer))))
    text_decoded = list(tokenizer.decode_batch(iter(text_encoded)))
    assert all(
        (sample[0] == tokenizer.BOS) & (sample[-1] == tokenizer.EOS)
        for sample in text_decoded
    )
    assert ["".join(sample[1:-1]) for sample in text_decoded] == text


@pytest.mark.parametrize(["n_jobs", "batch_size"], [(1, 2), (2, 2)])
def test_string_tokenizer(n_jobs: int, batch_size: int) -> None:
    text = [
        "1 2 3",
        "2 3 4",
    ]
    tokenizer = StringTokenizer(
        tokens=set(chain.from_iterable(text)) | {"2 3"},
        n_jobs=n_jobs,
        batch_size=batch_size,
    )

    assert (
        len(tokenizer) == 4 + 1 + 1 + 1 + 1
    )  # 4 digits + 1 bos + 1 eos + 1 space + 1x "2 3"
    text_encoded = list(tokenizer.encode_batch(iter(text)))
    assert set(np.unique(text_encoded)).issubset(set(range(len(tokenizer))))
    text_decoded = list(tokenizer.decode_batch(iter(text_encoded)))
    assert all(
        (sample[0] == tokenizer.BOS) & (sample[-1] == tokenizer.EOS)
        for sample in text_decoded
    )
    assert ["".join(sample[1:-1]) for sample in text_decoded] == text
    assert list(tokenizer.decode_batch(iter(text_encoded))) == [
        ["<bos>", "1", " ", "2 3", "<eos>"],
        ["<bos>", "2 3", " ", "4", "<eos>"],
    ]


@pytest.mark.parametrize(["n_jobs", "batch_size"], [(1, 2), (2, 2)])
def test_byte_pair_encoding_builder(n_jobs: int, batch_size: int) -> None:
    text = [
        "1 2 3 4",
        "2 3 4",
        "3",
        "2",
    ]
    tokenizer = BytePairEncodingBuilder(
        iter(text),
        num_tokens_max=None,
        n_jobs=n_jobs,
        batch_size=batch_size,
        verbose=True,
    ).get_tokenizer()
    assert tokenizer.tokens == {
        " ",
        "1",
        "2",
        "2 ",
        "2 3 ",
        "2 3 4<eos>",
        "3",
        "3 ",
        "4",
        "4<eos>",
        "<bos>",
        "<eos>",
    }
    text_encoded = list(tokenizer.encode_batch(iter(text)))
    assert set(chain.from_iterable([set(np.unique(t)) for t in text_encoded])).issubset(
        set(range(len(tokenizer)))
    )
    text_decoded = list(tokenizer.decode_batch(iter(text_encoded)))
    assert all(
        (sample[0] == tokenizer.BOS) & (sample[-1] == tokenizer.EOS)
        for sample in text_decoded
    )
    assert ["".join(sample[1:-1]) for sample in text_decoded] == text


@pytest.mark.parametrize(["n_jobs", "batch_size"], [(1, 2), (2, 2)])
@pytest.mark.parametrize(
    ["text", "tokens", "num_tokens_max"],
    [
        (
            ["1 2 3 4", "2 3 4"],
            {
                " ",
                "1",
                "2",
                "2 ",
                "2 3 ",
                "2 3 4<eos>",
                "3",
                "3 ",
                "4",
                "4<eos>",
                "<bos>",
                "<eos>",
            },
            None,
        ),
        (["abcdef"], {"a", "b", "c", "d", "e", "f", "<bos>", "<eos>"}, None),
        (
            ["abcabcdefgdefg"],
            {
                "<bos>",
                "<eos>",
                "a",
                "ab",
                "abc",
                "b",
                "c",
                "d",
                "de",
                "defg",
                "e",
                "f",
                "fg",
                "g",
            },
            None,
        ),
        (
            ["defgdefgabcabc"],
            {
                "<bos>",
                "<eos>",
                "a",
                "ab",
                "abc",
                "b",
                "c",
                "d",
                "de",
                "defg",
                "e",
                "f",
                "fg",
                "g",
            },
            None,
        ),
        (
            ["abcabcdede"],
            {
                "<bos>",
                "<eos>",
                "a",
                "ab",
                "abc",
                "b",
                "c",
                "d",
                "de",
                "e",
            },
            None,
        ),
        (
            ["abcabcdede"],
            {
                "<bos>",
                "<eos>",
                "a",
                "ab",
                # This is not a token because it exceeds the max number of tokens.
                # "abc",
                "b",
                "c",
                "d",
                "de",
                "e",
            },
            9,
        ),
        (
            ["abc"],
            {"<bos>", "<eos>", "a", "b", "c"},
            9,
        ),
    ],
)
def test_byte_pair_encoding_builder_on_data(
    n_jobs: int,
    batch_size: int,
    text: list[str],
    tokens: set[str],
    num_tokens_max: int | None,
) -> None:
    tokenizer = BytePairEncodingBuilder(
        iter(text), num_tokens_max=num_tokens_max, n_jobs=n_jobs, batch_size=batch_size
    ).get_tokenizer()
    assert tokenizer.tokens == tokens


@pytest.mark.parametrize(["n_jobs", "batch_size"], [(1, 2), (2, 2)])
def test_string_tokenizer_load_save(
    n_jobs: int, batch_size: int, tmp_path: Path
) -> None:
    text = [
        "1 2 3 4",
        "2 3 4",
    ]
    expected_tokens = {
        " ",
        "1",
        "2",
        "2 ",
        "2 3 ",
        "2 3 4<eos>",
        "3",
        "3 ",
        "4",
        "4<eos>",
        "<bos>",
        "<eos>",
    }
    tokenizer = BytePairEncodingBuilder(
        iter(text), num_tokens_max=None, n_jobs=n_jobs, batch_size=batch_size
    ).get_tokenizer()
    assert tokenizer.tokens == expected_tokens
    tokenizer.save(tmp_path)
    tokenizer_loaded = StringTokenizer(tokens=set())
    tokenizer_loaded.load(tmp_path)
    assert tokenizer_loaded.tokens == expected_tokens


@pytest.mark.parametrize(["n_jobs", "batch_size"], [(1, 2), (2, 2)])
def test_BPE_tokenizer_load_save(n_jobs: int, batch_size: int, tmp_path: Path) -> None:
    text = [
        "1 2 3 4",
        "2 3 4",
    ]
    expected_tokens = {
        " ",
        "1",
        "2",
        "2 ",
        "2 3 ",
        "2 3 4<eos>",
        "3",
        "3 ",
        "4",
        "4<eos>",
        "<bos>",
        "<eos>",
    }
    tokenizer = BytePairEncodingBuilder(
        iter(text), num_tokens_max=None, n_jobs=n_jobs, batch_size=batch_size
    ).get_tokenizer()
    assert tokenizer.tokens == expected_tokens
    tokenizer.save(tmp_path)
    tokenizer_loaded = StringTokenizer(tokens=set())
    tokenizer_loaded.load(tmp_path)
    assert tokenizer_loaded.tokens == expected_tokens


def test_replace_pair() -> None:
    sequences = [["1", " ", "2", " ", "3", " ", "4"]]

    # Initialize counts for expected sequence.
    counter: Counter[tuple[str, str]] = Counter()
    for sample in sequences:
        counter.update(pairwise(sample))

    sequences_updated, delta_counters = replace_pair(
        iter(sequences), pair_to_replace=(" ", "2"), new_token="haha"
    )
    assert list(sequences_updated) == [["1", "haha", " ", "3", " ", "4"]]

    # Initialize counts for expected sequence.
    counter = Counter()
    for sample in sequences:
        counter.update(pairwise(sample))

    # Initialize counts for expected sequence.
    counter_expected: Counter[tuple[str, str]] = Counter()
    for sample in sequences_updated:
        counter_expected.update(pairwise(sample))

    assert dict(delta_counters) == {
        ("1", "haha"): 1,
        ("haha", " "): 1,
        ("1", " "): -1,
        (" ", "2"): -1,
        ("2", " "): -1,
    }
    counter.update(dict(delta_counters))
    counter = +counter  # Remove zero elements.
    assert dict(counter) == dict(counter_expected)


@pytest.mark.parametrize(
    "size", [(28, 28, 3), (1, 28, 28, 3), (2, 28, 28, 3), (5, 28, 28, 3)]
)
def test_colored_mnist_image_tokenizer(size: tuple[int]) -> None:
    tokenizer = VQVAEImageTokenizer(vqvae=VQVAEMNIST(), verbose=True, batch_size=2)
    rng = np.random.default_rng(RANDOM_STATE)
    image = rng.choice(4, size=size, replace=True)
    image_encoded = tokenizer.encode(image, drop_bos=False, drop_eos=True)
    num_images = 1 if len(size) == 3 else size[0]
    assert len(image_encoded) == num_images
    assert all(len(im) == 7 * 7 + 1 for im in image_encoded)
    assert all(im[0] == tokenizer.bos_id for im in image_encoded)
    image_decoded = np.concatenate(
        list(tokenizer.decode_batch(iter(im[1:] for im in image_encoded)))
    )
    assert image_decoded.shape == (1, *size) if len(size) == 3 else size
