import warnings
from itertools import chain

from torchsmith.utils.dtypes import T


def chunk(sequence: list[T], sequence_length: int, warn: bool = False) -> list[list[T]]:
    num_complete_chunks = len(sequence) // sequence_length
    samples = sequence[: num_complete_chunks * sequence_length]

    # For the incomplete chunk, take the last `self.sequence_length` tokens.
    if len(sequence) % sequence_length != 0:
        samples += sequence[-sequence_length:]

    samples_as_list = [
        samples[i : i + sequence_length]
        for i in range(0, len(samples), sequence_length)
    ]
    if not all(len(sample) == sequence_length for sample in samples_as_list):
        if warn:
            offending_samples = [
                (len(sample), sample)
                for sample in samples_as_list
                if len(sample) != sequence_length
            ]
            _info = (
                f"Offending samples {len(offending_samples)}/{len(samples_as_list)}:\n"
                f"{offending_samples}"
            )
            warnings.warn(
                f"[SKIPPING] Not all samples have the correct length. \n{_info}",
                stacklevel=2,
            )
        samples_as_list = [
            sample for sample in samples_as_list if len(sample) == sequence_length
        ]
    return samples_as_list


def chunk_batch(
    sequences: list[list[T]], sequence_length: int, warn: bool = False
) -> list[list[T]]:
    return chunk(list(chain.from_iterable(sequences)), sequence_length, warn)


def replace_punctuation(text: str) -> str:
    replacements = {
        # Curly double quotes
        "“": '"',
        "”": '"',
        # Curly single quotes
        "‘": "'",  # noqa: RUF001
        "’": "'",  # noqa: RUF001
        # Dashes
        "—": "-",
        "–": "-",  # noqa: RUF001
        # Ellipsis
        "…": "...",
        # Thin spaces
        "\u2005": " ",
        "\u205f": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
