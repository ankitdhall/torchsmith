from collections.abc import Iterator

import numpy as np
import torch

from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.models.vae.base import BaseVQVAE
from torchsmith.tokenizers.base_tokenizer import BaseTokenizer
from torchsmith.utils.dtypes import TokenType
from torchsmith.utils.plotting import plot_images
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import batched

device = get_device()


class VQVAEImageTokenizer(BaseTokenizer[TokenType]):
    BOS = "<boi>"  # Beginning of image.
    EOS = "<eoi>"  # End of image.

    def __init__(
        self,
        *,
        vqvae: BaseVQVAE,
        n_jobs: int = 1,
        batch_size: int = 10000,
        verbose: bool = False,
        token_id_offset: int = 0,
    ) -> None:
        super().__init__(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)
        self.device = get_device()
        self.vqvae = vqvae.eval().to(self.device)
        self._id_to_token: dict[int, TokenType] = {
            (index + token_id_offset): token
            for index, token in enumerate(range(self.vqvae.codebook_size))
        }
        self._id_to_token.update(
            {max(self._id_to_token) + 1: self.BOS, max(self._id_to_token) + 2: self.EOS}
        )
        self._token_to_id = {token: idx for idx, token in self._id_to_token.items()}
        if self.verbose:
            print(
                f"Created a mapping for {len(self)} unique tokens: {self._token_to_id}"
            )

    @property
    def token_to_id(self) -> dict[TokenType, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> dict[int, TokenType]:
        return self._id_to_token

    def image_to_sequence(self, x: np.ndarray) -> np.ndarray:
        return (
            self.vqvae.encode_to_indices(torch.Tensor(x, device=device))
            .reshape(x.shape[0], -1)
            .cpu()
            .numpy()
        )

    @property
    def bos_id(self) -> int:
        return self.to_id(self.BOS)

    @property
    def eos_id(self) -> int:
        return self.to_id(self.EOS)

    @property
    def tokens(self) -> set[TokenType]:
        return set(self.token_to_id.keys())

    def encode(
        self, images: np.ndarray, drop_eos: bool, drop_bos: bool
    ) -> list[list[int]]:
        if not isinstance(images, np.ndarray):
            raise ValueError(
                f"Found '{type(images)}' for `image` instead of np.ndarray."
            )
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        sequences = []
        for batch in batched(images, self.batch_size):
            sequences.append(self.image_to_sequence(np.array(batch)))
        seq = np.concatenate(sequences, axis=0)
        map_func = np.vectorize(lambda word: self.token_to_id[word])
        seq_tokenized = map_func(seq)
        if not drop_bos or not drop_eos:
            rows = seq.shape[0]
            seq_parts = [seq]
            if not drop_bos:
                prefix = np.full((rows, 1), self.bos_id)
                seq_parts = [prefix, *seq_parts]
            if not drop_eos:
                suffix = np.full((rows, 1), self.eos_id)
                seq_parts = [*seq_parts, suffix]
            return np.concatenate(seq_parts, axis=1).tolist()
        else:
            return seq_tokenized.tolist()

    def decode(self, sequence: list[int]) -> np.ndarray:
        if not isinstance(sequence, list):
            raise ValueError(
                f"Found '{type(sequence)}' for `sequence` instead of list[int]."
            )
        sequence_2d = np.array(sequence).reshape(1, 7, 7)
        return (
            self.vqvae.decode_from_indices(torch.tensor(sequence_2d, device=device))
            .cpu()
            .numpy()
        )

    def decode_batch(self, sequences: Iterator[list[int]]) -> Iterator[np.ndarray]:
        if not isinstance(sequences, Iterator):
            raise ValueError(
                f"Found '{type(sequences)}' for `sequences` instead of "
                f"Iterator[list[int]]."
            )
        for t in sequences:
            yield self.decode(t)


def generate_samples_image(
    *,
    seq_len: int,
    tokenizer: VQVAEImageTokenizer,
    transformer: GPT2Decoder,
    decode: bool,
) -> torch.Tensor:
    transformer.eval()
    num_samples = 9  # TODO: expose this
    prefix = torch.full((num_samples, 1), tokenizer.bos_id, device=device, dtype=int)
    samples, _ = transformer.sample(
        num_samples, seq_len=seq_len, prefix=prefix, exclude_indices={tokenizer.bos_id}
    )
    if decode:
        # Skip the last token as it corresponds to what the model predicts after
        # the last pixel in the image. This is not required during decoding.
        # Skip the first token as it corresponds to the BOS token.
        # This is not required during decoding.
        decoded_images = list(
            tokenizer.decode_batch(iter(sample[1:] for sample in samples.tolist()))
        )
        plot_images(
            decoded_images, titles=[f"Sample {index}" for index in range(num_samples)]
        )
    return samples
