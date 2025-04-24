import huggingface_hub
import numpy as np
import pytest
import torch

from torchsmith.datahub.images_with_vqvae import ImagesWithVQVAEDataset
from torchsmith.datahub.svhn import postprocess_data
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.models.vae.vq_vae import VQVAE
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.tokenizers.vqvae_tokenizer import generate_samples_image_v2
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


@pytest.mark.parametrize(
    ["vqvae_path", "gpt2_path"],
    [
        ["ankitdhall/svhn_vqvae", "ankitdhall/svhn_gpt2"],
        ["ankitdhall/cifar10_vqvae", "ankitdhall/cifar10_gpt2"],
    ],
)
def test_vqvae_from_loaded_model(vqvae_path: str, gpt2_path: str) -> None:
    input_shape = (3, 32, 32)
    latent_shape = (32 // 4, 32 // 4)
    num_samples = 4
    rng = np.random.default_rng(seed=RANDOM_STATE)

    path_to_weights_vqvae = huggingface_hub.hf_hub_download(
        vqvae_path, filename="model.pth"
    )
    vqvae = VQVAE.load_model(path_to_weights_vqvae).to(device)
    vqvae_tokenizer = VQVAEImageTokenizer(vqvae=vqvae, batch_size=100)

    input_data = (
        rng.random((1, *input_shape)).astype("float32") - 0.5
    ) * 2  # [0, 1] -> [-1, 1]
    train_data_tokens = ImagesWithVQVAEDataset(input_data, tokenizer=vqvae_tokenizer)
    sequence_length = train_data_tokens.sequence_length
    assert sequence_length == np.prod(latent_shape).item() + 1

    path_to_weights_gpt2 = huggingface_hub.hf_hub_download(
        gpt2_path, filename="model.pth"
    )
    gpt2 = GPT2Decoder.load_model(path_to_weights_gpt2).to(device)

    with suppress_plot():
        samples = generate_samples_image_v2(
            seq_len=sequence_length,
            tokenizer=vqvae_tokenizer,
            transformer=gpt2,
            decode=True,
            num_samples=num_samples,
            postprocess_fn=postprocess_data,
        )

    assert samples.shape == torch.Size([num_samples, *input_shape])
