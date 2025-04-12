import json
from collections import Counter
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from itertools import chain
from itertools import pairwise
from itertools import tee
from pathlib import Path
from typing import Union
from typing import cast

from tqdm import tqdm

from torchsmith.tokenizers.string_tokenizer import StringTokenizer
from torchsmith.utils.dtypes import T
from torchsmith.utils.pyutils import batch_function


@dataclass
class MostCommonPair:
    pair: tuple[int, int]
    frequency: int
    decoded_string: str


@dataclass
class SymbolPair:
    symbol_id: int
    left: Union["SymbolPair", None]
    right: Union["SymbolPair", None]


def to_symbol_pairs(tokenized_text: Iterator[list[int]]) -> Iterator[SymbolPair]:
    raise NotImplementedError()


class BytePairEncodingTokenizer(StringTokenizer):
    def __init__(
        self,
        tokens: set[str],
        *,
        occurrence_count: Counter[tuple[int, int]],
        n_jobs: int = 1,
        batch_size: int = 1000,
        verbose: bool = False,
        token_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            tokens=tokens,
            token_counts=token_counts,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.occurrence_count = occurrence_count

    def save(self, path: str | Path) -> None:
        super().save(path)
        path_to_save = self.get_dir_to_save(Path(path))
        with open(path_to_save / "occurrence_count.json", "w") as f:
            counter_list = [
                {"key": pair, "value": count}
                for pair, count in dict(self.occurrence_count).items()
            ]
            json.dump(counter_list, f, indent=4)

    def load(self, path: str | Path) -> None:
        super().load(path)
        path_to_save = self.get_dir_to_save(Path(path))
        with open(path_to_save / "occurrence_count.json") as f:
            counter_list = json.load(f)
        counter_dict = {tuple(item["key"]): item["value"] for item in counter_list}
        # counter_dict = {tuple(pair): count for pair, count in d.items()}
        # counter_dict = {tuple(pair): count for pair, count in d.items()}
        self.occurrence_count = Counter(counter_dict)


# TODO: Pass a string tokenizer instead of inheriting it
class BytePairEncodingBuilder:
    def __init__(
        self,
        text: Iterator[str],
        *,
        num_tokens_max: int | None,
        batch_size: int = 1000,
        min_occurrences_to_merge: int = 2,
        verbose: bool = False,
        n_jobs: int = 1,
        save_interval_in_tokens: int | None = None,
        save_dir: Path | None = None,
        tokenizer_to_load_from: BytePairEncodingTokenizer | None = None,
    ) -> None:
        self.save_interval_in_tokens = save_interval_in_tokens
        self.save_dir = save_dir
        if (self.save_interval_in_tokens is not None and self.save_dir is None) or (
            self.save_interval_in_tokens is None and self.save_dir is not None
        ):
            raise ValueError(
                "Either both `save_interval_in_tokens` and `save_dir` "
                "should be provided or both not."
            )
        self.merged_token_current: MostCommonPair | None = None
        self.min_occurrences_to_merge = min_occurrences_to_merge
        self.num_tokens_max = num_tokens_max
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.batch_size = batch_size
        self.merged_tokens: list[MostCommonPair] = []

        if tokenizer_to_load_from is None:
            text_for_super, text_for_build = tee(text, 2)
            self._string_tokenizer = StringTokenizer(
                tokens=set(chain.from_iterable(text_for_super)),
                n_jobs=n_jobs,
                verbose=verbose,
                batch_size=batch_size,
            )
            self.occurrence_count: Counter[tuple[int, int]] = (
                Counter()
            )  # Counts tuples.
            self.bpe_tokenizer = self._build(text_for_build)
            self.tokens_found = 0
        else:
            self._string_tokenizer = tokenizer_to_load_from

            self.occurrence_count = self._string_tokenizer.occurrence_count
            self.tokens_found = len(self._string_tokenizer._str_to_count)
            self.bpe_tokenizer = self._build(text)

    def get_tokenizer(self) -> BytePairEncodingTokenizer:
        return self.bpe_tokenizer

    def _update_mappings(self, new_token: str, frequency: int) -> None:
        new_id = self._get_next_token_id
        self._string_tokenizer._id_to_str[new_id] = new_token
        self._string_tokenizer._str_to_id[new_token] = new_id
        self._string_tokenizer._str_to_count[new_token] = frequency

    @property
    def _get_next_token_id(self) -> int:
        return len(self._string_tokenizer._id_to_str) + len(self.merged_tokens)

    def _update_counter(
        self,
        text_and_counters: Iterator[
            tuple[list[int], Counter] | tuple[list[list[int]], Counter]
        ],
    ) -> Iterator[list[int]]:
        # TODO: Avoid storing text in memory
        texts: list[list[int]] = []
        for text, counter in text_and_counters:
            if len(text) != 0 and isinstance(text[0], list):
                text = cast(list[list[int]], text)
                texts.extend(text)
            else:
                text = cast(list[int], text)
                texts.append(text)
            self.occurrence_count.update(dict(counter))

        self.occurrence_count = +self.occurrence_count
        return (t for t in texts)

    def _prepare_count(self, tokenized_text: Iterator[list[int]]) -> None:
        if self.n_jobs == 1:
            for sample in tokenized_text:
                self.occurrence_count.update(pairwise(sample))
        else:
            counters: Iterator[Counter] = batch_function(
                func=update_occurrence_count,  # type: ignore
                input=tokenized_text,
                n_jobs=self.n_jobs,
                batch_size=self.batch_size,
            )
            for counter in counters:
                self.occurrence_count.update(counter)

    def _merge_tokens(self) -> bool:
        self._save_intermediate()

        if self.verbose:
            top_k = self.occurrence_count.most_common(5)
            top_k_str = "\n\t-".join(
                f"'{''.join(self._string_tokenizer.decode(list(pair)))}' "
                f"with freq. {freq}"
                for pair, freq in top_k
            )
            print("Top 5 pairs:\n\t-" + top_k_str)
        else:
            top_k = self.occurrence_count.most_common(1)

        most_common_pair, frequency = top_k[0]
        decoded_string = "".join(self._string_tokenizer.decode(list(most_common_pair)))

        # Nothing to do if:
        # (a) no pair occurs more than once or,
        # (b) the most commonly occurring pair is already a token.
        if (
            frequency < self.min_occurrences_to_merge
            or decoded_string in self._string_tokenizer.tokens
        ):
            _info = (
                f"frequency < self.min_occurrences_to_merge: {frequency} < "
                f"{self.min_occurrences_to_merge}\n"
                f"'{decoded_string}' in self._string_tokenizer.tokens: "
                f"{decoded_string in self._string_tokenizer.tokens}"
            )
            print(_info)
            return False

        if self.verbose:
            print(f"Adding new token: '{decoded_string}' with {frequency} occurrences")
        self.merged_token_current = MostCommonPair(
            pair=most_common_pair,
            frequency=frequency,
            decoded_string=decoded_string,
        )
        self.merged_tokens.append(self.merged_token_current)
        self._update_mappings(self.merged_token_current.decoded_string, frequency)
        return True

    def _save_intermediate(self, force_save: bool = False) -> None:
        if force_save or (
            self.save_interval_in_tokens is not None
            and len(self._string_tokenizer.tokens) % self.save_interval_in_tokens == 0
        ):
            intermediate_tokenize = BytePairEncodingTokenizer(
                tokens=self._string_tokenizer.tokens,
                token_counts=self._string_tokenizer._str_to_count,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                batch_size=self.batch_size,
                occurrence_count=self.occurrence_count,
            )
            assert self.save_dir is not None
            intermediate_tokenize.save(
                self.save_dir / f"{len(self._string_tokenizer.tokens)}"
            )

    def _build(self, text: Iterator[str]) -> BytePairEncodingTokenizer:
        tokenized_text: Iterator[list[int]] = self._string_tokenizer.encode_batch(text)
        # TODO: Avoid calling tee, instead call func twice?
        tokenized_text_to_merge, tokenized_text = tee(tokenized_text, 2)
        tokenized_text_to_merge = cast(Iterator[list[int]], tokenized_text_to_merge)
        self._prepare_count(tokenized_text_to_merge)

        with tqdm(
            total=self.num_tokens_max or 10000, desc="Merging tokens ..."
        ) as pbar:
            pbar.update(len(self._string_tokenizer.tokens))
            while (
                len(self._string_tokenizer.tokens) < self.num_tokens_max
                if self.num_tokens_max is not None
                else True
            ) and self._merge_tokens():
                assert self.merged_token_current is not None
                func = partial(
                    replace_pair,
                    pair_to_replace=self.merged_token_current.pair,
                    new_token=self._string_tokenizer._str_to_id[
                        self.merged_token_current.decoded_string
                    ],
                )
                text_and_counters: Iterator[tuple[list[int], Counter]] = batch_function(
                    func=func,  # type: ignore
                    input=tokenized_text,
                    n_jobs=self.n_jobs,
                    batch_size=self.batch_size,
                )
                tokenized_text = self._update_counter(text_and_counters)
                pbar.update(1)
            print(
                f"Found {len(self._string_tokenizer.tokens)}/"
                f"{self.num_tokens_max} tokens"
            )

        if self.save_dir is not None:
            self._save_intermediate(force_save=True)
        return BytePairEncodingTokenizer(
            tokens=self._string_tokenizer.tokens,
            token_counts=self._string_tokenizer._str_to_count,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            batch_size=self.batch_size,
            occurrence_count=self.occurrence_count,
        )


def update_occurrence_count(xs: list[list[int]]) -> Counter:
    counter: Counter = Counter()
    for x in xs:
        counter.update(pairwise(x))
    return counter


def replace_pair(
    sequences: Iterator[list[T]], *, pair_to_replace: tuple[T, T], new_token: T
) -> tuple[list[list[T]], Counter[tuple[T, T]]]:
    # TODO: can this be done in places
    assert isinstance(pair_to_replace, tuple) and len(pair_to_replace) == 2
    sequences_updated = []
    count_dict: dict[tuple[T, T], int] = defaultdict(int)
    for sequence in sequences:
        sequence_updated: list[T] = []
        pair = None
        skip_next = False
        for pair in pairwise(sequence):
            if skip_next:
                skip_next = False
                count_dict[(new_token, pair[1])] += 1
                count_dict[(pair_to_replace[1], pair[1])] -= 1
                continue

            if pair == pair_to_replace:
                count_dict[pair_to_replace] -= 1
                if len(sequence_updated) != 0:
                    count_dict[(sequence_updated[-1], new_token)] += 1
                    count_dict[(sequence_updated[-1], pair_to_replace[0])] -= 1
                skip_next = True
                sequence_updated.append(new_token)
            else:
                sequence_updated.append(pair[0])
        if pair is not None and not skip_next:
            sequence_updated.append(pair[-1])
        sequences_updated.append(sequence_updated)
    delta_count = Counter(count_dict)
    return sequences_updated, delta_count
