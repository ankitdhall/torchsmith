import huggingface_hub
import torch

from torchsmith.models.external._vqvae import VQVAE
from torchsmith.utils.pytorch import get_device


def load_pretrain_vqvae() -> VQVAE:
    path_to_args = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_vqvae", filename="vqvae_args_colored_mnist_2_ft.pth"
    )
    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/colored_mnist_vqvae", filename="vqvae_colored_mnist_2_ft.pth"
    )
    loaded_args = torch.load(path_to_args)
    vqvae = VQVAE(**loaded_args)
    vqvae.load_state_dict(torch.load(path_to_weights, map_location=get_device()))
    return vqvae
