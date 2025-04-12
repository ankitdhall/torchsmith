import huggingface_hub
import numpy as np
import pytest
import torch

from torchsmith.datahub.colored_mnist import ColoredMNISTWithTextDataset
from torchsmith.datahub.colored_mnist import load_colored_mnist_dataset
from torchsmith.models.gpt2.decoder import GPT2Decoder
from torchsmith.tokenizers import CharacterTokenizer
from torchsmith.tokenizers.mnist_tokenizer import ColoredMNISTImageAndTextTokenizer
from torchsmith.tokenizers.mnist_tokenizer import VQVAEColoredMNISTImageTokenizer
from torchsmith.tokenizers.mnist_tokenizer import (
    colored_mnist_with_text_conditioned_on_image,
)
from torchsmith.tokenizers.mnist_tokenizer import (
    colored_mnist_with_text_conditioned_on_text,
)
from torchsmith.tokenizers.mnist_tokenizer import generate_samples_colored_mnist_image
from torchsmith.tokenizers.mnist_tokenizer import (
    generate_samples_colored_mnist_with_text,
)
from torchsmith.tokenizers.mnist_tokenizer import sample_completion_image
from torchsmith.tokenizers.text_tokenizer import sample_completion_text
from torchsmith.training.config import GPT2Config
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import plot_images
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


def test_gpt2_text_sample() -> None:
    sequence_length = 5
    num_samples = 2
    text = [
        "Let the bird of loudest lay\r\n"
        "On the sole Arabian tree\r\n"
        "Herald sad and trumpet be"
    ]
    tokenizer = CharacterTokenizer.from_text(iter(text))

    transformer_config = GPT2Config(seq_len=sequence_length)
    transformer = GPT2Decoder.from_config(
        vocab_size=len(tokenizer),
        config=transformer_config,
    ).to(device)

    prefix = torch.full((num_samples, 1), tokenizer.bos_id, device=device, dtype=int)
    samples, sampling_times = transformer.sample(
        num_samples, prefix=prefix, seq_len=sequence_length
    )
    assert samples.shape == torch.Size([num_samples, sequence_length])
    assert set(samples.unique().tolist()).issubset(set(range(len(tokenizer))))
    assert sampling_times.shape == (sequence_length,)


def test_gpt2_text_sample_completion() -> None:
    sequence_length = 10
    text = [
        "Let the bird of loudest lay\r\n"
        "On the sole Arabian tree\r\n"
        "Herald sad and trumpet be"
    ]
    tokenizer = CharacterTokenizer.from_text(iter(text))

    transformer_config = GPT2Config(seq_len=sequence_length)
    transformer = GPT2Decoder.from_config(
        vocab_size=len(tokenizer),
        config=transformer_config,
    ).to(device)

    prefixes_text = ["Let the", "On the", "Herald"]
    prefixes_encoded = list(
        tokenizer.encode_batch(iter(prefixes_text), drop_bos=True, drop_eos=True)
    )
    for prefix_encoded, prefix_text in zip(prefixes_encoded, prefixes_text):
        prefix = torch.tensor(
            [tokenizer.bos_id, *prefix_encoded], device=device, dtype=int
        ).unsqueeze(0)
        samples, _ = transformer.sample(1, prefix=prefix, seq_len=sequence_length)
        generated_tokens = samples[0].tolist()[1:]  # Drop <bos> token at pos 0.
        assert all(a == b for a, b in (zip(prefix_encoded, generated_tokens)))
        generated_tokens_text = tokenizer.decode(generated_tokens)
        generated_sample = "".join(generated_tokens_text)
        assert generated_sample.startswith(prefix_text)
        assert samples.shape == torch.Size([1, sequence_length])
        assert set(samples.unique().tolist()).issubset(set(range(len(tokenizer))))


def test_sample_completion_text() -> None:
    sequence_length = 10
    text = [
        "Let the bird of loudest lay\r\n"
        "On the sole Arabian tree\r\n"
        "Herald sad and trumpet be"
    ]
    tokenizer = CharacterTokenizer.from_text(iter(text))

    transformer_config = GPT2Config(seq_len=sequence_length)
    transformer = GPT2Decoder.from_config(
        vocab_size=len(tokenizer),
        config=transformer_config,
    ).to(device)

    prefixes_text = ["Let the", "Let the", "Let the", "On the", "Herald"]
    generated_text = sample_completion_text(
        prefixes=prefixes_text,
        tokenizer=tokenizer,
        model=transformer,
        skip_indices={tokenizer.bos_id, tokenizer.eos_id},
    )
    for prefix, generated in zip(prefixes_text, generated_text):
        assert generated.startswith(prefix)


def test_gpt2_colored_mnist_sample() -> None:
    num_samples = 9
    tokenizer = VQVAEColoredMNISTImageTokenizer(batch_size=10000)
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_gpt2", filename="model.pth"
    )
    transformer = GPT2Decoder.load_model(path_to_weights).to(device)
    sequence_length = transformer.seq_len
    with suppress_plot():
        samples = generate_samples_colored_mnist_image(
            seq_len=sequence_length,
            tokenizer=tokenizer,
            transformer=transformer,
            decode=True,
        )
    assert samples.shape == torch.Size([num_samples, sequence_length])
    assert set(samples.unique().tolist()).issubset(set(range(len(tokenizer))))


def test_sample_completion_colored_mnist() -> None:
    num_completions = 4
    tokenizer = VQVAEColoredMNISTImageTokenizer(batch_size=10000)
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_gpt2", filename="model.pth"
    )
    transformer = GPT2Decoder.load_model(path_to_weights).to(device)

    images, _ = load_colored_mnist_dataset("test")
    images = images[40:45]
    with suppress_plot():
        samples = sample_completion_image(
            images=images,
            show_fraction=0.35,
            num_completions=num_completions,
            tokenizer=tokenizer,
            model=transformer,
            skip_indices={tokenizer.bos_id},
        )
    assert len(samples) == len(images) * (num_completions + 1)


def test_gpt2_colored_mnist_with_text_sample() -> None:
    tokenizer = ColoredMNISTImageAndTextTokenizer()
    transformer_config = GPT2Config(seq_len=ColoredMNISTWithTextDataset.SEQUENCE_LENGTH)
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_with_text_gpt2", filename="model.pth"
    )
    transformer = GPT2Decoder.load_model(path_to_weights).to(device)
    with suppress_plot():
        generate_samples_colored_mnist_with_text(
            seq_len=transformer_config.seq_len,
            tokenizer=tokenizer,
            transformer=transformer,
            decode=True,
            num_samples=16,
        )


@pytest.mark.parametrize(
    ["text", "num_samples"],
    [
        ("plain orange seven on dark blue", 4),
        ("dark blue seven on plain orange", 4),
    ],
)
def test_gpt2_colored_mnist_with_text_sample_completion_conditioned_on_text(
    text: str, num_samples: int
) -> None:
    tokenizer = ColoredMNISTImageAndTextTokenizer()
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_with_text_gpt2", filename="model.pth"
    )
    transformer = GPT2Decoder.load_model(path_to_weights).to(device)

    with suppress_plot():
        decoded_images, decoded_texts = colored_mnist_with_text_conditioned_on_text(
            num_samples=num_samples,
            text=text,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        plot_images(
            decoded_images, titles=decoded_texts, max_cols=int(np.sqrt(num_samples))
        )


DIGITS = ["one", "two", "three", "four", "five", "six", "seven", "eigth", "nine"]


@pytest.mark.parametrize(
    "input_texts",
    [
        [f"dark red {digit} on light cyan" for digit in DIGITS],
        [f"plain red {digit}" for digit in DIGITS],
        ["plain red" for _ in DIGITS],
    ],
)
def test_gpt2_colored_mnist_with_text_sample_completion_conditioned_on_text_2(
    input_texts: list[str],
) -> None:
    tokenizer = ColoredMNISTImageAndTextTokenizer()
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_with_text_gpt2", filename="model.pth"
    )
    transformer = GPT2Decoder.load_model(path_to_weights).to(device)
    num_samples = len(input_texts)
    images, texts = [], []
    for text in input_texts:
        decoded_images, decoded_texts = colored_mnist_with_text_conditioned_on_text(
            num_samples=1,
            text=text,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        images.append(decoded_images[0])
        texts.append(decoded_texts[0])
    with suppress_plot():
        plot_images(images, titles=texts, max_cols=int(np.sqrt(num_samples)))


def test_colored_mnist_with_text_sample_image_completion() -> None:
    num_completions = 4
    tokenizer = ColoredMNISTImageAndTextTokenizer()
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_with_text_gpt2", filename="model.pth"
    )
    transformer = GPT2Decoder.load_model(path_to_weights).to(device)
    images, _ = load_colored_mnist_dataset("test")
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_indices = rng.choice(len(images), size=5, replace=False)
    images = images[sampled_indices]
    with suppress_plot():
        samples, texts = colored_mnist_with_text_conditioned_on_image(
            images=images,
            show_fraction=0.35,
            num_completions=num_completions,
            tokenizer=tokenizer,
            model=transformer,
        )
    assert len(samples) == len(texts) == len(images) * (num_completions + 1)
