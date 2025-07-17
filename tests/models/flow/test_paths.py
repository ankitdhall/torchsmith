from typing import Literal

import pytest
from matplotlib import pyplot as plt

from torchsmith.models.flow.data import Gaussian
from torchsmith.models.flow.data import GaussianMixture
from torchsmith.models.flow.paths.gaussian_conditional_probability_path import (
    GaussianConditionalProbabilityPath,
)
from torchsmith.models.flow.paths.schedulers import LinearAlpha
from torchsmith.models.flow.paths.schedulers import SquareRootBeta
from torchsmith.models.flow.visualize import (
    visualize_conditional_probability_path_overlaid,
)
from torchsmith.models.flow.visualize import (
    visualize_conditional_probability_trajectories,
)
from torchsmith.models.flow.visualize import visualize_density
from torchsmith.utils.plotting import suppress_plot
from torchsmith.utils.pytorch import get_device

device = get_device()


def test_visualize_conditional_probability_path() -> None:
    target_scale = 10.0
    target_std = 1.0
    plot_limits = 15.0
    p_source = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p_data = GaussianMixture.symmetric_2d(
        num_modes=5, std=target_std, scale=target_scale
    ).to(device)
    path = GaussianConditionalProbabilityPath(
        p_source=p_source, p_target=p_data, alpha=LinearAlpha(), beta=SquareRootBeta()
    ).to(device)
    z = path.sample_conditioning_variable(1)

    with suppress_plot():
        ax = visualize_conditional_probability_path_overlaid(
            path=path, z=z, plot_limits=plot_limits
        )
        visualize_density(
            p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=ax
        )
        plt.show()


@pytest.mark.parametrize("mode", ["ode", "sde"])
def test_visualize_trajectories(mode: Literal["ode", "sde"]) -> None:
    target_scale = 10.0
    target_std = 1.0
    plot_limits = 15.0
    p_source = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p_data = GaussianMixture.symmetric_2d(
        num_modes=5, std=target_std, scale=target_scale
    ).to(device)
    path = GaussianConditionalProbabilityPath(
        p_source=p_source, p_target=p_data, alpha=LinearAlpha(), beta=SquareRootBeta()
    ).to(device)
    z = path.sample_conditioning_variable(1)
    with suppress_plot():
        fig, axes = plt.subplots(1, 2, figsize=(36, 12))
        visualize_density(
            p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=axes[0]
        )
        visualize_conditional_probability_path_overlaid(
            path=path, z=z, plot_limits=plot_limits, ax=axes[0]
        )
        visualize_density(
            p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=axes[1]
        )
        visualize_conditional_probability_trajectories(
            path=path, z=z, plot_limits=plot_limits, ax=axes[1], mode=mode
        )
        plt.show()
