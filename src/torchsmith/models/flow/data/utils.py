from typing import Optional

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from torchsmith.models.flow.data import Density
from torchsmith.models.flow.data import Sampleable
from torchsmith.utils.pytorch import get_device

device = get_device()


def hist2d_sampleable(
    sampleable: Sampleable, *, num_samples: int, ax: Optional[Axes] = None, **kwargs
) -> None:
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)
    ax.hist2d(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def scatter_sampleable(
    sampleable: Sampleable, *, num_samples: int, ax: Optional[Axes] = None, **kwargs
) -> None:
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)
    ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def plot_density(
    density: Density,
    *,
    show_contours: bool,
    bins: int,
    scale: float,
    ax: Optional[Axes] = None,
    density_kwargs: dict | None = None,
    contours_kwargs: dict | None = None,
) -> None:
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    log_density = density.log_density(xy).reshape(bins, bins).T
    ax.imshow(
        log_density.cpu(),
        extent=(-scale, scale, -scale, scale),
        origin="lower",
        **(density_kwargs or {}),
    )
    if show_contours:
        ax.contour(
            log_density.cpu(),
            extent=(-scale, scale, -scale, scale),
            origin="lower",
            **(contours_kwargs or {}),
        )


def visualize_densities(
    densities: dict[str, Density], bins: int = 100, scale: float = 15
) -> None:
    fig, axes = plt.subplots(1, len(densities), figsize=(6 * len(densities), 6))
    for idx, (name, density) in enumerate(densities.items()):
        ax = axes[idx]
        ax.set_title(name)
        plot_density(
            density,
            bins=bins,
            scale=scale,
            ax=ax,
            show_contours=True,
            density_kwargs=dict(vmin=-15, cmap=plt.get_cmap("Blues")),
            contours_kwargs=dict(
                colors="grey", linestyles="solid", alpha=0.25, levels=20
            ),
        )
    plt.show()
