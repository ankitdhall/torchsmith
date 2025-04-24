from typing import cast

import numpy as np
import torch
from tqdm import tqdm

from torchsmith.datahub.colored_mnist import ColoredMNISTWithTextDataset
from torchsmith.datahub.colored_mnist import get_unique_tokens
from torchsmith.models.external.colored_mnist import load_pretrain_vqvae
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.models.vae import BaseVQVAE
from torchsmith.tokenizers.base_tokenizer import BaseTokenizer
from torchsmith.tokenizers.string_tokenizer import WordTokenizer
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.utils.dtypes import TokenType
from torchsmith.utils.plotting import darken_fraction_of_image
from torchsmith.utils.plotting import plot_images
from torchsmith.utils.pytorch import get_device

device = get_device()


class VQVAEMNIST(BaseVQVAE):
    def __init__(self) -> None:
        super().__init__()
        self.vqvae = load_pretrain_vqvae().eval().to(device)

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, self.vqvae.quantize(x.numpy()).reshape(x.shape[0], -1)
        )

    def decode_from_indices(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.tensor(self.vqvae.decode(x.numpy()))

    @property
    def codebook_size(self) -> int:
        return self.vqvae.n_embeddings


class ColoredMNISTImageAndTextTokenizer(BaseTokenizer):
    BOS = "<start>"

    def __init__(
        self,
        *,
        n_jobs: int = 1,
        batch_size: int = 1000,
        verbose: bool = False,
        token_id_offset: int = 0,
    ) -> None:
        super().__init__(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)
        self.device = get_device()

        text_tokens = get_unique_tokens()
        self.image_tokenizer = VQVAEImageTokenizer(
            vqvae=VQVAEMNIST(), token_id_offset=token_id_offset
        )
        self.text_tokenizer = WordTokenizer(
            tokens=text_tokens,
            verbose=True,
            token_id_offset=token_id_offset + len(self.image_tokenizer),
        )

        self._id_to_token: dict[int, TokenType] = {}
        self._id_to_token.update(self.image_tokenizer.id_to_token)
        self._id_to_token.update(self.text_tokenizer.id_to_token)
        self._id_to_token.update({max(self._id_to_token) + 1: self.BOS})

        self._token_to_id = {token: idx for idx, token in self._id_to_token.items()}
        if set(self._token_to_id) != set(self.image_tokenizer.tokens) | {
            self.BOS
        } | set(self.text_tokenizer.tokens):
            _a = set(self._token_to_id)
            _b = (
                set(self.text_tokenizer.tokens)
                | set(self.image_tokenizer.tokens)
                | {self.BOS}
            )
            raise ValueError(f"{_a} \n != \n {_b}")
        if self.verbose:
            print(
                f"Created a mapping for {len(self)} unique tokens: {self._token_to_id}"
            )

    def tokenize_text(self, text: list[str]) -> torch.Tensor:
        return torch.tensor(
            list(
                self.text_tokenizer.encode_batch(
                    iter(text), drop_bos=False, drop_eos=False
                )
            )
        )

    def tokenize_images(self, images: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            self.image_tokenizer.encode(images, drop_bos=False, drop_eos=False)
        )

    @property
    def token_to_id(self) -> dict[TokenType, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> dict[int, TokenType]:
        return self._id_to_token

    @property
    def bos_id(self) -> int:
        return self.to_id(self.BOS)

    @property
    def tokens(self) -> set[TokenType]:
        return set(self.token_to_id.keys())


def generate_samples_colored_mnist_with_text(
    *,
    seq_len: int,
    tokenizer: ColoredMNISTImageAndTextTokenizer,
    transformer: GPT2Decoder,
    decode: bool,
    num_samples: int = 9,
) -> tuple[list[np.ndarray], list[str]]:
    transformer.eval()
    samples_complete = torch.full((num_samples, seq_len), -1, device=device, dtype=int)

    prefix = torch.full((num_samples, 1), tokenizer.bos_id, device=device, dtype=int)
    samples_all, _ = transformer.sample(
        num_samples,
        seq_len=1 + 1,  # <start> and text OR image BOS.
        prefix=prefix,
        include_indices={
            tokenizer.image_tokenizer.bos_id,
            tokenizer.text_tokenizer.bos_id,
        },
    )
    is_text_bos = samples_all[:, 1] == tokenizer.text_tokenizer.bos_id

    # Part 1.
    num_samples_text_first = int(is_text_bos.sum().item())
    samples_text_first = samples_all[is_text_bos]

    samples_text_first = sample_text_then_image(
        num_samples=num_samples_text_first,
        prefix=samples_text_first,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    samples_complete[is_text_bos] = samples_text_first

    # Part 2.
    samples_image_first = samples_all[~is_text_bos]
    num_samples_image_first = len(samples_image_first)

    samples_image_first = sample_image_then_text(
        num_samples=num_samples_image_first,
        prefix=samples_image_first,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    samples_complete[~is_text_bos] = samples_image_first

    sample_images, sample_text = separate_images_and_text(
        is_text_bos=is_text_bos,
        num_samples=num_samples,
        samples_image_first=samples_image_first,
        samples_text_first=samples_text_first,
    )

    if decode:
        # Skip the last token as it corresponds to what the model predicts after
        # the last pixel in the image. This is not required during decoding.
        # Skip the first token as it corresponds to the BOS token.
        # This is not required during decoding.
        decoded_images, decoded_texts = decode_images_and_text(
            tokenizer=tokenizer, sample_images=sample_images, sample_text=sample_text
        )
        plot_images(
            decoded_images, titles=decoded_texts, max_cols=int(num_samples**0.5)
        )
    return decoded_images, decoded_texts


def decode_images_and_text(
    *,
    tokenizer: ColoredMNISTImageAndTextTokenizer,
    sample_images: torch.Tensor,
    sample_text: torch.Tensor,
) -> tuple[list[np.ndarray], list[str]]:
    decoded_images = list(
        tokenizer.image_tokenizer.decode_batch(iter(sample_images.tolist()))
    )
    decoded_texts = [
        " ".join(texts)
        for texts in list(
            tokenizer.text_tokenizer.decode_batch(iter(sample_text.tolist()))
        )
    ]
    return decoded_images, decoded_texts


def separate_images_and_text(
    *,
    is_text_bos: torch.Tensor,
    num_samples: int,
    samples_image_first: torch.Tensor,
    samples_text_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert is_text_bos.shape[0] == num_samples
    assert is_text_bos.sum().item() == samples_text_first.shape[0]
    assert samples_image_first.shape[0] + samples_text_first.shape[0] == num_samples

    # Separate images and text.
    sample_images = torch.zeros(
        (num_samples, ColoredMNISTWithTextDataset.IMAGE_LENGTH - 2),
        dtype=torch.long,
        device=device,
    )
    sample_images[is_text_bos] = samples_text_first[
        :,
        (1 + ColoredMNISTWithTextDataset.TEXT_LENGTH + 1) : (
            1 + ColoredMNISTWithTextDataset.TEXT_LENGTH + 1
        )
        + (ColoredMNISTWithTextDataset.IMAGE_LENGTH - 2),
    ]
    sample_images[~is_text_bos] = samples_image_first[
        :,
        2 : 2 + (ColoredMNISTWithTextDataset.IMAGE_LENGTH - 2),
    ]
    sample_text = torch.zeros(
        (num_samples, ColoredMNISTWithTextDataset.TEXT_LENGTH - 2),
        dtype=torch.long,
        device=device,
    )
    sample_text[~is_text_bos] = samples_image_first[
        :,
        (1 + ColoredMNISTWithTextDataset.IMAGE_LENGTH + 1) : (
            1 + ColoredMNISTWithTextDataset.IMAGE_LENGTH + 1
        )
        + (ColoredMNISTWithTextDataset.TEXT_LENGTH - 2),
    ]
    sample_text[is_text_bos] = samples_text_first[
        :, 2 : 2 + (ColoredMNISTWithTextDataset.TEXT_LENGTH - 2)
    ]
    return sample_images, sample_text


def sample_image_then_text(
    *,
    num_samples: int,
    prefix: torch.Tensor,
    tokenizer: ColoredMNISTImageAndTextTokenizer,
    transformer: GPT2Decoder,
) -> torch.Tensor:
    token_ids_text = {
        tokenizer.token_to_id[token] for token in tokenizer.text_tokenizer.tokens
    }
    token_ids_image = {
        tokenizer.token_to_id[token] for token in tokenizer.image_tokenizer.tokens
    }
    # Sample image.
    samples, _ = transformer.sample(
        num_samples,
        seq_len=1
        + (ColoredMNISTWithTextDataset.IMAGE_LENGTH - 1),  # Omit the image EOS.
        prefix=prefix,
        include_indices=token_ids_image
        - {tokenizer.image_tokenizer.bos_id, tokenizer.image_tokenizer.eos_id},
    )
    # Append image EOS and text BOS.
    samples = torch.cat(
        [
            samples,
            torch.full(
                (num_samples, 1),
                tokenizer.image_tokenizer.eos_id,
                device=samples.device,
            ),
            torch.full(
                (num_samples, 1), tokenizer.text_tokenizer.bos_id, device=samples.device
            ),
        ],
        dim=-1,
    )
    # Sample text.
    samples, _ = transformer.sample(
        num_samples,
        seq_len=1
        + ColoredMNISTWithTextDataset.IMAGE_LENGTH
        + (ColoredMNISTWithTextDataset.TEXT_LENGTH - 1),  # Omit the text EOS.
        prefix=samples,
        include_indices=token_ids_text
        - {tokenizer.text_tokenizer.bos_id, tokenizer.text_tokenizer.eos_id},
    )
    # Append text EOS.
    samples = torch.cat(
        [
            samples,
            torch.full(
                (num_samples, 1), tokenizer.text_tokenizer.eos_id, device=samples.device
            ),
        ],
        dim=-1,
    )
    return samples


def sample_text_then_image(
    *,
    num_samples: int,
    prefix: torch.Tensor,
    tokenizer: ColoredMNISTImageAndTextTokenizer,
    transformer: GPT2Decoder,
) -> torch.Tensor:
    token_ids_text = {
        tokenizer.token_to_id[token] for token in tokenizer.text_tokenizer.tokens
    }
    token_ids_image = {
        tokenizer.token_to_id[token] for token in tokenizer.image_tokenizer.tokens
    }
    # Sample text.
    samples, _ = transformer.sample(
        num_samples,
        seq_len=1 + (ColoredMNISTWithTextDataset.TEXT_LENGTH - 1),  # Omit the text EOS.
        prefix=prefix,
        include_indices=token_ids_text
        - {tokenizer.text_tokenizer.bos_id, tokenizer.text_tokenizer.eos_id},
    )
    # Append text EOS and image BOS.
    samples = torch.cat(
        [
            samples,
            torch.full(
                (num_samples, 1), tokenizer.text_tokenizer.eos_id, device=samples.device
            ),
            torch.full(
                (num_samples, 1),
                tokenizer.image_tokenizer.bos_id,
                device=samples.device,
            ),
        ],
        dim=-1,
    )
    # Sample image.
    samples, _ = transformer.sample(
        num_samples,
        seq_len=1
        + ColoredMNISTWithTextDataset.TEXT_LENGTH
        + (ColoredMNISTWithTextDataset.IMAGE_LENGTH - 1),  # Omit the image EOS.
        prefix=samples,
        include_indices=token_ids_image
        - {tokenizer.image_tokenizer.bos_id, tokenizer.image_tokenizer.eos_id},
    )
    # Append image EOS.
    samples = torch.cat(
        [
            samples,
            torch.full(
                (num_samples, 1),
                tokenizer.image_tokenizer.eos_id,
                device=samples.device,
            ),
        ],
        dim=-1,
    )
    return samples


def sample_completion_image(
    *,
    images: np.ndarray,
    show_fraction: float,
    num_completions: int,
    tokenizer: VQVAEImageTokenizer,
    model: GPT2Decoder,
    skip_indices: set[int] | None = None,
) -> list[np.ndarray]:
    generated_samples, titles = [], []
    index = 1
    prefixes_encoded = tokenizer.encode(images, drop_bos=True, drop_eos=True)
    boundary_index = int(len(prefixes_encoded[0]) * show_fraction)
    for image, prefix_encoded in tqdm(
        zip(images, prefixes_encoded), "Completing image ..."
    ):
        prefix = (
            torch.tensor(
                [tokenizer.bos_id, *prefix_encoded[:boundary_index]],
                device=device,
                dtype=int,
            )
            .unsqueeze(0)
            .repeat(num_completions, 1)
        )
        samples, _ = model.sample(
            num_completions,
            prefix=prefix,
            seq_len=model.seq_len,
            exclude_indices=skip_indices,
        )  # (B, (1 + H*W))
        samples = samples[:, 1:]  # Drop <bos> token at pos 0.
        generated_tokens = samples.tolist()
        generated_images = list(tokenizer.decode_batch(iter(generated_tokens)))
        generated_samples.extend(
            [darken_fraction_of_image(image, show_fraction, 0.8), *generated_images]
        )
        titles.extend(
            [
                f"Image {index}",
                *[
                    f"Image {index} Completion {completion_index + 1}"
                    for completion_index in range(num_completions)
                ],
            ]
        )
        index += 1
    plot_images(generated_samples, titles=titles, max_cols=num_completions + 1)
    return generated_samples


def colored_mnist_with_text_conditioned_on_text(
    *,
    num_samples: int,
    text: str,
    tokenizer: ColoredMNISTImageAndTextTokenizer,
    transformer: GPT2Decoder,
) -> tuple[list[np.ndarray], list[str]]:
    prefix_texts = torch.tensor(
        tokenizer.text_tokenizer.encode(text, drop_bos=False, drop_eos=True),
        device=device,
    ).repeat((num_samples, 1))

    prefix = torch.cat(
        [
            torch.full((num_samples, 1), tokenizer.bos_id, device=device, dtype=int),
            prefix_texts,
        ],
        dim=-1,
    )
    samples_text_first = sample_text_then_image(
        num_samples=num_samples,
        prefix=prefix,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    sample_images, sample_text = separate_images_and_text(
        is_text_bos=torch.full([num_samples], True, device=device),
        num_samples=num_samples,
        samples_image_first=torch.empty(
            [0, samples_text_first.shape[1]], dtype=torch.long, device=device
        ),
        samples_text_first=samples_text_first,
    )
    decoded_images, decoded_texts = decode_images_and_text(
        tokenizer=tokenizer, sample_images=sample_images, sample_text=sample_text
    )

    return decoded_images, decoded_texts


def colored_mnist_with_text_conditioned_on_image(
    *,
    tokenizer: ColoredMNISTImageAndTextTokenizer,
    images: np.ndarray,
    show_fraction: float,
    num_completions: int,
    model: GPT2Decoder,
) -> tuple[list[np.ndarray], list[str]]:
    generated_samples, titles = [], []
    index = 1
    prefixes_encoded = tokenizer.image_tokenizer.encode(
        images, drop_bos=True, drop_eos=True
    )
    boundary_index = int(len(prefixes_encoded[0]) * show_fraction)

    for image, prefix_encoded in tqdm(
        zip(images, prefixes_encoded), "Completing image ..."
    ):
        prefix = (
            torch.tensor(
                [
                    tokenizer.bos_id,
                    tokenizer.image_tokenizer.bos_id,
                    *prefix_encoded[:boundary_index],
                ],
                device=device,
                dtype=int,
            )
            .unsqueeze(0)
            .repeat(num_completions, 1)
        )
        samples_image_first = sample_image_then_text(
            num_samples=num_completions,
            prefix=prefix,
            tokenizer=tokenizer,
            transformer=model,
        )
        sample_images, sample_text = separate_images_and_text(
            is_text_bos=torch.full([num_completions], False, device=device),
            num_samples=num_completions,
            samples_text_first=torch.empty(
                [0, samples_image_first.shape[1]], dtype=torch.long, device=device
            ),
            samples_image_first=samples_image_first,
        )
        _images, _texts = decode_images_and_text(
            tokenizer=tokenizer, sample_images=sample_images, sample_text=sample_text
        )
        generated_samples.extend(
            [darken_fraction_of_image(image, show_fraction, 0.8), *_images]
        )
        titles.extend(
            [
                f"Image {index}",
                *[
                    _texts[completion_index]
                    for completion_index in range(num_completions)
                ],
            ]
        )
        index += 1

    plot_images(
        generated_samples,
        titles=titles,
        max_cols=num_completions + 1,
        main_title="Image Completion Followed by Text Generation",
    )
    return generated_samples, titles
