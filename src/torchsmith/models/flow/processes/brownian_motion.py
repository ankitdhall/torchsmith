import torch
from matplotlib import pyplot as plt

from torchsmith.models.flow.processes import SDE
from torchsmith.models.flow.solvers.euler_maruyama import EulerMaruyamaSolver
from torchsmith.utils.pytorch import get_device

device = get_device()


class BrownianMotion(SDE):
    """Brownian Motion.
    Brownian motion is recovered (by definition) by setting
    :math:`u_t = 0` and :math:`\\sigma_t = \\sigma`.

    i.e.
    .. math::
        dX_t = \\sigma dW_t, \\quad X_0 = x_0.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x_t, 0)

    def diffusion_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x_t, self.sigma)


def plot_brownian_motion_trajectories(process_parameters: list[dict]) -> None:
    start_time = 0.0
    end_time = 20.0
    num_timesteps = 1000

    x0 = torch.zeros(10).view(-1, 1).to(device)
    ts = torch.linspace(start_time, end_time, num_timesteps).to(device)

    fig, axes = plt.subplots(
        1, len(process_parameters), figsize=(5 * len(process_parameters), 5)
    )
    for index, parameters in enumerate(process_parameters):
        process = BrownianMotion(**parameters)
        em_solver = EulerMaruyamaSolver(process)
        trajectories = em_solver.simulate_trajectories(x0, ts)

        ax = axes[index]
        ax.set_title(
            f"Trajectories of Brownian Motion with $\\sigma = ${process.sigma}",
            fontsize=15,
        )
        ax.set_xlabel("Time ($t$)", fontsize=15)
        ax.set_ylabel("$X_t$", fontsize=15)

        for trajectory in trajectories:
            ax.plot(ts.cpu().numpy(), trajectory.cpu().numpy(), alpha=0.75)

    plt.show()
