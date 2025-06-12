import contextlib
from pathlib import Path
from typing import Literal

import pytest
import torch

from torchsmith.models.flow.data import Gaussian
from torchsmith.models.flow.data import GaussianMixture
from torchsmith.models.flow.data.utils import visualize_densities
from torchsmith.models.flow.data.utils import visualize_trajectories
from torchsmith.models.flow.processes import LangevinDynamics
from torchsmith.models.flow.solvers import EulerMaruyamaSolver
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


@pytest.mark.parametrize(
    ["save_as", "filename", "is_valid"],
    [
        ("movie", "trajectories.mp4", True),
        ("movie", "trajectories.gif", True),
        ("movie", None, False),
        ("image", "trajectories.png", True),
        ("image", None, True),
    ],
)
def test_visualize_trajectories(
    save_as: Literal["movie", "image"], filename: str, is_valid: bool, tmp_path: Path
) -> None:
    source_distribution = Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(
        device
    )
    target_distribution = GaussianMixture.random_2d(
        num_modes=5, std=0.75, scale=15.0, seed=3
    ).to(device)
    sde = LangevinDynamics(sigma=10.0, density=target_distribution)
    solver = EulerMaruyamaSolver(sde)

    context = contextlib.nullcontext() if is_valid else pytest.raises(ValueError)
    with suppress_plot(), context:
        visualize_trajectories(
            num_samples=1000,
            source_distribution=source_distribution,
            solver=solver,
            density=target_distribution,
            timesteps=torch.linspace(0, 5.0, 1000).to(device),
            plot_every=100,
            save_as=save_as,
            save_path=(tmp_path / filename) if filename is not None else None,
        )
    if filename is not None and is_valid:
        assert (tmp_path / filename).exists()
