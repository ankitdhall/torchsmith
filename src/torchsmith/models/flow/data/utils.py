from typing import Optional

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from torchsmith.models.flow.data import Density
from torchsmith.models.flow.data import Sampleable
from torchsmith.models.flow.solvers import Solver
from torchsmith.utils.pytorch import get_device

device = get_device()


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
    """Plots the log-density and (optionally) contours for a 2D density object.

    Parameters:
        density: The density object to visualize.
        show_contours: Whether to overlay contour lines on the plot.
        bins: Number of bins for the grid in each dimension. The grid is used to
            evaluate and plot the probability density.
        scale: [-scale, +scale] is the extent of the plot in for both x-axis and y-axis.
        ax: Matplotlib Axes to plot on. If None, uses current axes.
        density_kwargs: Additional keyword arguments for the density image.
        contours_kwargs: Additional keyword arguments for the contour lines.
    """
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
    """Visualizes contours and probability density of 2D densities side by side.

    Args:
        densities: Dictionary mapping names to Density objects.
        bins: Number of bins for the density plot.
        scale: X-axis and y-axis extent for the plots. Extent is [-scale, +scale].
    """
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


def visualize_trajectories(
    num_samples: int,
    *,
    source_distribution: Sampleable,
    solver: Solver,
    density: Density,
    timesteps: torch.Tensor,
    plot_every: int,
    bins: int = 200,
) -> None:
    """Visualizes the trajectories of the SDE/ODE solver.

    Solves the DE with the x_0 coming from the source distribution.
    The solver then evolves the x_0 through the timesteps.
    The evolution of the trajectories are governed by:
        (1) the drift for ODEs and,
        (2) the drift and diffusion for SDEs.

    Args:
        num_samples: Number of data points to evolve using the solver from t=0 to t=1.
        source_distribution: The initial distribution to sample points from.
        solver: The solver used to simulate the trajectories.
        density: The target density.
        timesteps: 1D tensor of time points for simulation.
        plot_every: Interval for selecting timesteps to plot.
        bins: Number of bins for density plots.
    """

    x_0 = source_distribution.sample(num_samples)
    xts = solver.simulate_trajectories(x_0, timesteps)
    scale = float(xts.max().abs().item())
    indices_to_plot = torch.cat(
        (
            torch.arange(0, len(timesteps) - 1, step=plot_every),
            torch.tensor([len(timesteps) - 1]),
        )
    )
    plot_timesteps = timesteps[indices_to_plot]

    fig, axes = plt.subplots(
        2, len(plot_timesteps), figsize=(8 * len(plot_timesteps), 16)
    )
    axes = axes.reshape((2, len(plot_timesteps)))
    for t_idx in range(len(plot_timesteps)):
        t = plot_timesteps[t_idx].item()
        x_t = xts[:, t_idx]

        # Step 1: Plot scatter x_t and target density.
        scatter_ax = axes[0, t_idx]
        plot_density(
            density,
            show_contours=False,
            bins=bins,
            scale=scale,
            ax=scatter_ax,
            density_kwargs=dict(vmin=-15, alpha=0.25, cmap=plt.get_cmap("Blues")),
        )
        scatter_ax.scatter(
            x_t[:, 0].cpu(),
            x_t[:, 1].cpu(),
            marker="x",
            color="black",
            alpha=0.75,
            s=15,
        )
        scatter_ax.set_title(f"Samples at t={t:.1f}", fontsize=15)
        scatter_ax.set_xticks([])
        scatter_ax.set_yticks([])

        # Step 2: Plot contours using x_t and target density.
        kdeplot_ax = axes[1, t_idx]
        plot_density(
            density,
            show_contours=False,
            bins=bins,
            scale=scale,
            ax=kdeplot_ax,
            density_kwargs=dict(vmin=-15, alpha=0.25, cmap=plt.get_cmap("Blues")),
        )
        sns.kdeplot(
            x=x_t[:, 0].cpu(), y=x_t[:, 1].cpu(), alpha=0.5, ax=kdeplot_ax, color="grey"
        )
        kdeplot_ax.set_title(f"Density of Samples at t={t:.1f}", fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")

    plt.show()
