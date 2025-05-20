import torch

from torchsmith.models.flow.data import Gaussian
from torchsmith.models.flow.data import GaussianMixture
from torchsmith.models.flow.data.utils import visualize_densities
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


def test_densities() -> None:
    with suppress_plot():
        visualize_densities(
            {
                "Gaussian": Gaussian(mean=torch.zeros(2), cov=10 * torch.eye(2)).to(
                    device
                ),
                "Random Mixture (seed = 3)": GaussianMixture.random_2d(
                    num_modes=5, std=1.0, scale=20.0, seed=3
                ).to(device),
                "Random Mixture (seed = 4)": GaussianMixture.random_2d(
                    num_modes=5, std=1.0, scale=20.0, seed=4
                ).to(device),
                "Symmetric Mixture": GaussianMixture.symmetric_2d(
                    num_modes=5, std=1.0, scale=8.0
                ).to(device),
            }
        )
