from typing import Literal
from typing import Optional

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from torchsmith.models.flow.data.base import Density
from torchsmith.models.flow.data.base import Sampleable
from torchsmith.models.flow.data.base import SampleableDensity
from torchsmith.models.flow.data.utils import plot_density
from torchsmith.models.flow.paths.conditional_probability_path import (
    ConditionalProbabilityPath,
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
) -> Axes:
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
    return ax


def visualize_conditional_probability_path_overlaid(
    path: ConditionalProbabilityPath,
    *,
    z: torch.Tensor,
    plot_limits: float,
    num_samples: int = 1000,
    ax: Axes | None = None,
) -> Axes:
    assert z.shape[0] == 1, "z must be a single conditioning variable (1, dim)"
    if ax is None:
        ax = plt.gca()

    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_title("Conditional Probability Path (Ground Truth)", fontsize=20)

    timesteps = torch.linspace(0.0, 1.0, 7).to(device)

    # Plot z
    ax.scatter(z[:, 0].cpu(), z[:, 1].cpu(), marker="*", color="red", s=80, label="z")
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot conditional probability path at each intermediate t
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


def visualize_conditional_probability_path(
    path: ConditionalProbabilityPath,
    *,
    z: torch.Tensor,
    plot_limits: float,
    num_time_intervals: int = 5,
    num_samples: int = 1000,
) -> Axes:
    assert z.shape[0] == 1, "z must be a single conditioning variable (1, dim)"
    fig, ax = plt.subplots(1, num_time_intervals, figsize=(3 * num_time_intervals, 3))
    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]

    fig.suptitle("Conditional Probability Path (Ground Truth)", fontsize=20)

    timesteps = torch.linspace(0.0, 1.0, num_time_intervals).to(device)

    # Plot conditional probability path at each intermediate t
    for index in range(timesteps.shape[0]):
        t = timesteps[index]
        z_batch = z.expand(num_samples, 2)
        t_batch = t.unsqueeze(0).expand(num_samples, 1)  # (num_samples, 1)
        samples = path.sample_conditional_path(z_batch, t_batch)  # (num_samples, 2)
        ax[index].scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            alpha=0.25,
            s=8,
        )
        ax[index].set_title(f"t={t.item():.1f}")
        ax[index].set_xlim(*x_bounds)
        ax[index].set_ylim(*y_bounds)
        ax[index].set_xticks([])
        ax[index].set_yticks([])

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
    return ax


def visualize_generated_trajectories(
    solver: Solver,
    *,
    p_source: Sampleable,
    plot_limits: float,
    num_trajectories: int = 15,
    num_timesteps: int = 1000,
    ax: Optional[Axes] = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    if isinstance(solver, EulerSolver):
        mode = "ode"
    elif isinstance(solver, EulerMaruyamaSolver):
        mode = "sde"
    else:
        raise ValueError(f"Invalid solver: {solver}")

    x_0 = p_source.sample(num_trajectories)  # (num_samples, 2)
    timesteps = torch.linspace(0.0, 1.0, num_timesteps, device=device)
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

    for trajectory_index in range(num_trajectories):
        ax.plot(
            x_t[trajectory_index, :, 0].detach().cpu(),
            x_t[trajectory_index, :, 1].detach().cpu(),
            alpha=0.5,
            color="black",
        )
    return ax


def visualize_marginal_probability_path_overlaid(
    path: ConditionalProbabilityPath,
    *,
    num_samples: int,
    plot_limits: float,
    num_time_intervals: int = 7,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_title("Gaussian Marginal Probability Path", fontsize=20)

    timesteps = torch.linspace(0.0, 1.0, num_time_intervals).to(device)

    ax.set_xticks([])
    ax.set_yticks([])

    # Plot marginal probability path at each intermediate t
    for t in timesteps:
        marginal_samples = path.sample_marginal_path(  # (num_samples, 1)
            t.expand(num_samples, 1)
        )
        ax.scatter(
            marginal_samples[:, 0].cpu(),
            marginal_samples[:, 1].cpu(),
            alpha=0.25,
            s=8,
            label=f"t={t.item():.1f}",
        )

    ax.legend(prop={"size": 18}, markerscale=3)
    return ax


def visualize_marginal_probability_path(
    path: ConditionalProbabilityPath,
    *,
    num_samples: int,
    plot_limits: float,
    num_time_intervals: int = 5,
) -> Axes:
    fig, ax = plt.subplots(1, num_time_intervals, figsize=(3 * num_time_intervals, 3))
    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]

    fig.suptitle("Marginal Probability Path (Ground Truth)", fontsize=20)

    timesteps = torch.linspace(0.0, 1.0, num_time_intervals).to(device)

    # Plot marginal probability path at each intermediate t
    for index in range(timesteps.shape[0]):
        t = timesteps[index]
        marginal_samples = path.sample_marginal_path(  # (num_samples, 1)
            t.expand(num_samples, 1)
        )
        ax[index].scatter(
            marginal_samples[:, 0].cpu(),
            marginal_samples[:, 1].cpu(),
            alpha=0.25,
            s=8,
        )
        ax[index].set_title(f"t={t.item():.1f}")
        ax[index].set_xlim(*x_bounds)
        ax[index].set_ylim(*y_bounds)
        ax[index].set_xticks([])
        ax[index].set_yticks([])

    return ax


def visualize_samples_from_learned_marginal_overlaid(
    path: ConditionalProbabilityPath,
    *,
    solver: Solver,
    plot_limits: float,
    num_samples: int = 1000,
    num_time_intervals: int = 7,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    solver_type = "SDE" if isinstance(solver, EulerMaruyamaSolver) else "ODE"
    ax.set_title(f"Samples from Learned Marginal {solver_type}", fontsize=20)

    timesteps = torch.linspace(0.0, 1.0, 1000).to(device)

    # Construct integrator and plot trajectories
    x0 = path.p_source.sample(num_samples)  # (num_samples, 2)
    xts = solver.simulate_trajectories(x0, timesteps)  # (bs, nts, dim)

    for t_index in range(
        0, timesteps.shape[0], timesteps.shape[0] // (num_time_intervals - 1)
    ):
        t = timesteps[t_index]
        ax.scatter(
            xts[:, t_index, 0].cpu(),
            xts[:, t_index, 1].cpu(),
            alpha=0.25,
            s=8,
            label=f"t={t.item():.1f}",
        )

    ax.legend(prop={"size": 18}, markerscale=3)
    return ax


def visualize_samples_from_learned_marginal(
    path: ConditionalProbabilityPath,
    *,
    solver: Solver,
    plot_limits: float,
    num_samples: int = 1000,
    num_time_intervals: int = 5,
) -> Axes:
    fig, ax = plt.subplots(1, num_time_intervals, figsize=(3 * num_time_intervals, 3))
    x_bounds = [-plot_limits, plot_limits]
    y_bounds = [-plot_limits, plot_limits]

    solver_type = "SDE" if isinstance(solver, EulerMaruyamaSolver) else "ODE"
    fig.suptitle(f"Samples from Learned Marginal {solver_type}", fontsize=20)

    timesteps = torch.linspace(0.0, 1.0, 1000).to(device)

    # Construct integrator and plot trajectories
    x0 = path.p_source.sample(num_samples)  # (num_samples, 2)
    xts = solver.simulate_trajectories(x0, timesteps)  # (bs, nts, dim)

    for plot_index, index in enumerate(
        range(0, timesteps.shape[0], timesteps.shape[0] // num_time_intervals)
    ):
        t = timesteps[index]
        ax[plot_index].scatter(
            xts[:, index, 0].cpu(),
            xts[:, index, 1].cpu(),
            alpha=0.25,
            s=8,
        )
        ax[plot_index].set_title(f"t={t.item():.1f}")
        ax[plot_index].set_xlim(*x_bounds)
        ax[plot_index].set_ylim(*y_bounds)
        ax[plot_index].set_xticks([])
        ax[plot_index].set_yticks([])

    return ax


def visualize_field_across_time_and_space(
    model: torch.nn.Module,
    *,
    path: ConditionalProbabilityPath,
    num_marginals: int,
    plot_limits: float,
    num_bins: int,
    title: str,
) -> Axes:
    fig, axes = plt.subplots(1, num_marginals, figsize=(6 * num_marginals, 6))

    timesteps = torch.linspace(0.0, 1.0, num_marginals).to(device)
    xs = torch.linspace(-plot_limits, plot_limits, num_bins).to(device)
    ys = torch.linspace(-plot_limits, plot_limits, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=-1)

    for t_index in range(num_marginals):
        t = timesteps[t_index]
        batch_size = num_bins**2
        tt = t.view(1, 1).expand(batch_size, 1)

        # Learned vector field/scores
        learned_field = model(xy, tt)
        learned_field_x = learned_field[:, 0]
        learned_field_y = learned_field[:, 1]

        ax = axes[t_index]
        ax.quiver(
            xx.detach().cpu(),
            yy.detach().cpu(),
            learned_field_x.detach().cpu(),
            learned_field_y.detach().cpu(),
            scale=125,
            alpha=0.5,
        )

        if isinstance(path.p_source, Density) and isinstance(path.p_target, Density):
            ax = visualize_density(
                p_source=path.p_source,  # type: ignore
                p_data=path.p_target,  # type: ignore
                plot_limits=plot_limits,
                ax=ax,
            )
        ax.set_title(f"At t={t.item():.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=20)
    return axes
