import torch

from torchsmith.models.vae.vq_vae import VQVAE


def test_vqvae():
    input_shape = (3, 32, 32)
    latent_shape = (32 // 4, 32 // 4)
    latent_dim = 16
    num_samples = 5
    codebook_size = 10
    vqvae = VQVAE(input_shape, latent_dim=latent_dim, codebook_size=codebook_size)
    input = (torch.rand((num_samples, *input_shape)) - 0.5) * 2  # [0, 1] -> [-1, 1]
    x_reconstructed = vqvae(input)
    assert x_reconstructed.shape == torch.Size([num_samples, *input_shape])

    input_encoded = vqvae.encode(input)
    input_decoded = vqvae.decode(input_encoded)
    assert torch.isin(input_encoded, torch.arange(0, codebook_size, 1)).all()
    assert input_encoded.shape == torch.Size([num_samples, *latent_shape])
    assert input_decoded.shape == torch.Size([num_samples, latent_dim, *latent_shape])

    loss_total, loss_reconstruction, loss_codebook_encoder = vqvae.loss(input)
    assert loss_total == loss_reconstruction + loss_codebook_encoder
