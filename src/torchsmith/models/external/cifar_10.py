import huggingface_hub
import torch

from torchsmith.models.external._vae import VAE
from torchsmith.utils.pytorch import get_device


def load_pretrain_vqvae() -> VAE:
    vqvae = VAE()

    path_to_weights = huggingface_hub.hf_hub_download(
        "ankitdhall/cifar_10_vae", filename="vae_cifar10.pth"
    )
    vqvae.load_state_dict(torch.load(path_to_weights, map_location=get_device()))
    vqvae.eval()
    return vqvae