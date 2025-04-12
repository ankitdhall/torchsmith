from abc import abstractmethod
from collections.abc import Iterator
from functools import partial

import torch
from tqdm import tqdm

from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.tokenizers.base_tokenizer import BaseTokenizer
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import batch_function

device = get_device()


class TextTokenizer(BaseTokenizer[str]):
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(
        self,
        *,
        n_jobs: int = 1,
        batch_size: int = 1000,
        verbose: bool = False,
    ) -> None:
        super().__init__(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)

    @abstractmethod
    def split_text(self, x: str) -> list[str]:
        raise NotImplementedError()

    @property
    def bos_id(self) -> int:
        return self.to_id(self.BOS)

    @property
    def eos_id(self) -> int:
        return self.to_id(self.EOS)

    def encode(
        self, text: str, *, drop_eos: bool = False, drop_bos: bool = False
    ) -> list[int]:
        if not isinstance(text, str):
            raise ValueError(f"Found '{type(text)}' for `text` instead of string.")
        if drop_bos and drop_eos:
            sequence = [*self.split_text(text)]
        else:
            sequence = [*self.split_text(text)]
            if not drop_bos:
                sequence = [self.BOS, *sequence]
            if not drop_eos:
                sequence = [*sequence, self.EOS]
        return list(map(lambda word: self.token_to_id[word], sequence))

    def _encode_batch(
        self, text: list[str], *, drop_eos: bool = False, drop_bos: bool = False
    ) -> list[list[int]]:
        if not isinstance(text, list):
            raise ValueError(f"Found '{type(text)}' for `text` instead of list[str].")
        return [self.encode(t, drop_eos=drop_eos, drop_bos=drop_bos) for t in text]

    def encode_batch(
        self, text: Iterator[str], *, drop_eos: bool = False, drop_bos: bool = False
    ) -> Iterator[list[int]]:
        if not isinstance(text, Iterator):
            raise ValueError(
                f"Found '{type(text)}' for `text` instead of Iterator[str]."
            )

        if self.n_jobs != 1:
            results: Iterator[list[list[int]]] = batch_function(
                func=partial(self._encode_batch, drop_eos=drop_eos, drop_bos=drop_bos),  # type: ignore
                input=text,
                n_jobs=self.n_jobs,
                batch_size=self.batch_size,
            )
            for encoded_batch_text in results:
                yield from encoded_batch_text
        else:
            for t in text:
                yield self.encode(t, drop_eos=drop_eos, drop_bos=drop_bos)

    def decode(self, text: list[int]) -> list[str]:
        if not isinstance(text, list):
            raise ValueError(f"Found '{type(text)}' for `text` instead of list[int].")
        return list(map(lambda idx: self.id_to_token[idx], text))  # type: ignore

    def decode_batch(self, text: Iterator[list[int]]) -> Iterator[list[str]]:
        if not isinstance(text, Iterator):
            raise ValueError(
                f"Found '{type(text)}' for `text` instead of Iterator[list[int]."
            )
        for t in text:
            yield self.decode(t)


def generate_samples_text(
    *,
    seq_len: int,
    tokenizer: TextTokenizer,
    transformer: GPT2Decoder,
    decode: bool,
) -> torch.Tensor:
    transformer.eval()
    num_samples = 5
    prefix = torch.full((num_samples, 1), tokenizer.bos_id, device=device, dtype=int)
    samples, _ = transformer.sample(num_samples, seq_len=seq_len, prefix=prefix)
    if decode:
        decoded_text = list(tokenizer.decode_batch(iter(samples.tolist())))
        for index, text in enumerate(decoded_text):
            print(f"------ Start of Sample {index} ------")
            print("".join(text))
            print(f"------ End of Sample {index} ------")
    return samples


def sample_completion_text(
    *,
    prefixes: list[str],
    tokenizer: TextTokenizer,
    model: GPT2Decoder,
    skip_indices: set[int] | None = None,
) -> list[str]:
    generated_samples = []
    prefixes_encoded = list(
        tokenizer.encode_batch(iter(prefixes), drop_bos=True, drop_eos=True)
    )
    for prefix_encoded in tqdm(prefixes_encoded, "Completing text ..."):
        prefix = torch.tensor(
            [tokenizer.bos_id, *prefix_encoded], device=device, dtype=int
        ).unsqueeze(0)
        samples, _ = model.sample(
            1, prefix=prefix, seq_len=model.seq_len, exclude_indices=skip_indices
        )
        generated_tokens = samples[0].tolist()[1:]  # Drop <bos> token at pos 0.
        generated_tokens_text = tokenizer.decode(generated_tokens)
        generated_sample = "".join(generated_tokens_text)
        generated_samples.append(generated_sample)
    print(generated_samples)
    return generated_samples
