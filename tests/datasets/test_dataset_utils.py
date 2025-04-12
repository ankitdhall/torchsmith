import warnings

import pytest

from torchsmith.datahub.utils import chunk
from torchsmith.datahub.utils import chunk_batch
from torchsmith.utils.dtypes import T


@pytest.mark.parametrize(
    ["sequence", "sequence_length", "expected"],
    [
        # Case 1: Sequence length is exactly divisible by sequence_length.
        ([1, 2, 3, 4, 5, 6], 3, [[1, 2, 3], [4, 5, 6]]),
        # Case 2: Sequence not exactly divisible by sequence_length.
        # The last (incomplete) chunk is completed by taking the last sequence_length
        # tokens.
        ([1, 2, 3, 4, 5], 3, [[1, 2, 3], [3, 4, 5]]),
        # Case 3: Sequence length exactly equals sequence_length.
        ([1, 2, 3], 3, [[1, 2, 3]]),
        # Case 4: Test with non-numeric data.
        (["a", "b", "c", "d", "e"], 2, [["a", "b"], ["c", "d"], ["d", "e"]]),
        # Case 5: Empty sequence should return an empty list.
        ([], 3, []),
    ],
)
def test_chunk(sequence: list[T], sequence_length: int, expected: list[T]) -> None:
    result = chunk(sequence, sequence_length)
    assert result == expected


@pytest.mark.parametrize(
    ["sequences", "sequence_length", "expected", "expect_warning"],
    [
        # Test when sequences list is empty.
        ([], 3, [], False),
        # Test when overall tokens are fewer than sequence_length.
        ([[1, 2]], 3, [], True),
        # Test when overall tokens exactly fill complete chunks.
        ([[1, 2, 3], [4, 5, 6]], 3, [[1, 2, 3], [4, 5, 6]], False),
        # Test when there are leftover tokens that need to be filled from the next
        # sample.
        ([[1, 2, 3], [4, 5]], 3, [[1, 2, 3], [3, 4, 5]], False),
        # Test with multiple sequences.
        (
            [[1, 2], [3, 4, 5], [6, 7, 8, 9]],
            3,
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            False,
        ),
        # Test with non-numeric data.
        ([["a", "b"], ["c", "d", "e"]], 3, [["a", "b", "c"], ["c", "d", "e"]], False),
    ],
)
def test_chunk_batch(
    sequences: list[list[T]],
    sequence_length: int,
    expected: list[T],
    expect_warning: bool,
) -> None:
    result = chunk_batch(sequences, sequence_length, warn=True)
    print(result)
    if expect_warning:
        with pytest.warns(UserWarning, match=r"\[SKIPPING\] Not all samples"):
            result = chunk_batch(sequences, sequence_length, warn=True)
    else:
        with warnings.catch_warnings(record=True) as warning_info:
            result = chunk_batch(sequences, sequence_length, warn=True)
        assert len(warning_info) == 0

    assert result == expected
