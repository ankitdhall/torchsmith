from torchsmith.models.flow.processes import plot_brownian_motion_trajectories
from torchsmith.models.flow.processes import plot_ou_process_trajectories
from torchsmith.utils.plotting import suppress_plot


def test_brownian_motion() -> None:
    with suppress_plot():
        plot_brownian_motion_trajectories([dict(sigma=0.25), dict(sigma=1.0)])


def test_ou_process() -> None:
    with suppress_plot():
        plot_ou_process_trajectories(
            [dict(theta=0.25, sigma=0.0), dict(theta=0.25, sigma=0.25)]
        )
