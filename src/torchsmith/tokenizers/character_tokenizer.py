from collections.abc import Iterator
from itertools import chain

import pandas as pd

from torchsmith.tokenizers.text_tokenizer import TextTokenizer


class CharacterTokenizer(TextTokenizer):
    def __init__(
        self,
        tokens: set[str],
        *,
        n_jobs: int = 1,
        batch_size: int = 1000,
        verbose: bool = False,
    ) -> None:
        super().__init__(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)
        tokens = tokens.union({self.BOS, self.EOS})
        self.id_to_char = dict(enumerate(sorted(tokens)))
        self.char_to_id = {char: idx for idx, char in self.id_to_char.items()}

        if self.verbose:
            print(f"Created a mapping for {len(self)} unique tokens: {self.char_to_id}")

    @property
    def token_to_id(self) -> dict[str, int]:
        return self.char_to_id

    @property
    def id_to_token(self) -> dict[int, str]:
        return self.id_to_char

    def split_text(self, x: str) -> list[str]:
        assert isinstance(x, str), f"Found {type(x)} for {x} instead of string."
        return list(x)

    @classmethod
    def from_df(cls, df: pd.DataFrame, column: str, **kwargs) -> "CharacterTokenizer":
        unique_tokens = set(
            chain.from_iterable(set(value) for index, value in df[column].items())
        )
        return CharacterTokenizer(unique_tokens, **kwargs)

    @classmethod
    def from_text(cls, text: Iterator[str], **kwargs) -> "CharacterTokenizer":
        unique_tokens = set(chain.from_iterable(text))
        return CharacterTokenizer(unique_tokens, **kwargs)
