from typing import Literal
from typing import Optional

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from torchsmith.models.flow.data.base import SampleableDensity
from torchsmith.models.flow.data.utils import plot_density
from torchsmith.models.flow.paths.conditional_probability_path import (
    ConditionalProbabilityPath,
)
from torchsmith.models.flow.paths.gaussian_conditional_probability_path import (
    GaussianConditionalProbabilityPath,
)
from torchsmith.models.flow.processes.conditional_vector_field import (
    ConditionalVectorFieldODE,
)
from torchsmith.models.flow.processes.conditional_vector_field import (
    ConditionalVectorFieldSDE,
)
from torchsmith.models.flow.solvers import EulerMaruyamaSolver
from torchsmith.models.flow.solvers import EulerSolver
from torchsmith.models.flow.solvers import Solver
from torchsmith.utils.pytorch import get_device

device = get_device()


def visualize_density(
    p_source: SampleableDensity,
    *,
    p_data: SampleableDensity,
    plot_limits: float,
    ax: Axes | None = None,
):
    if ax is None:
        ax = plt.gca()

    # Plot source and target
    plot_density(
        density=p_source,
        show_contours=False,
        scale=plot_limits,
        bins=200,
        ax=ax,
        density_kwargs=dict(vmin=-15, alpha=0.25, cmap=plt.get_cmap("Reds")),
    )
    plot_density(
        density=p_data,
        show_contours=False,
        scale=plot_limits,
        bins=200,
        ax=ax,
        density_kwargs=dict(vmin=-15, alpha=0.25, cmap=plt.get_cmap("Blues")),
    )


def visualize_conditional_probability_path(
    path: GaussianConditionalProbabilityPath,
    *,
    z: torch.Tensor,
    plot_limits: float,
    ax: Axes | None = None,
) -> Axes:
    assert z.shape[0] == 1, "z must be a single conditioning variable (1, dim)"
    if ax is None:
        ax = plt.gca()

    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_title("Gaussian Conditional Probability Path")

    timesteps = torch.linspace(0.0, 1.0, 7).to(device)

    # Plot z
    ax.scatter(z[:, 0].cpu(), z[:, 1].cpu(), marker="*", color="red", s=80, label="z")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot conditional probability path at each intermediate t
    num_samples = 1000
    for t in timesteps:
        z_batch = z.expand(num_samples, 2)
        t_batch = t.unsqueeze(0).expand(num_samples, 1)  # (num_samples, 1)
        samples = path.sample_conditional_path(z_batch, t_batch)  # (num_samples, 2)
        ax.scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            alpha=0.25,
            s=8,
            label=f"t={t.item():.1f}",
        )

    ax.legend(prop={"size": 18}, markerscale=3)
    return ax


def visualize_conditional_probability_trajectories(
    path: ConditionalProbabilityPath,
    *,
    mode: Literal["ode", "sde"],
    z: torch.Tensor,
    plot_limits: float,
    num_trajectories: int = 15,
    num_timesteps: int = 1000,
    ax: Optional[Axes] = None,
    sigma: float = 2.5,  # Only used for SDE mode.
) -> Axes:
    if ax is None:
        ax = plt.gca()

    if mode == "ode":
        ode = ConditionalVectorFieldODE(path, z)
        solver: Solver = EulerSolver(ode)
    elif mode == "sde":
        sde = ConditionalVectorFieldSDE(path, z=z, sigma=sigma)
        solver = EulerMaruyamaSolver(sde)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'ode' or 'sde'.")

    x_0 = path.p_source.sample(num_trajectories)  # (num_samples, 2)
    timesteps = torch.linspace(
        0.0, 1.0, num_timesteps, device=device
    )  # (num_timesteps,)
    x_t = solver.simulate_trajectories(
        x_0, timesteps
    )  # (num_samples, num_timesteps, dim)

    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Trajectories of Conditional {mode.upper()}", fontsize=20)
    ax.scatter(
        z[:, 0].cpu(),
        z[:, 1].cpu(),
        marker="*",
        color="red",
        s=200,
        label="z",
        zorder=20,
    )

    for trajectory_index in range(num_trajectories):
        ax.plot(
            x_t[trajectory_index, :, 0].detach().cpu(),
            x_t[trajectory_index, :, 1].detach().cpu(),
            alpha=0.5,
            color="black",
        )
    ax.legend(prop={"size": 24}, loc="upper right", markerscale=1.8)
    return ax
