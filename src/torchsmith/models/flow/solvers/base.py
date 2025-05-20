from abc import ABC
from abc import abstractmethod

import torch
from tqdm import tqdm


class Solver(ABC):
    @abstractmethod
    def step(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Returns :math:`x_{t + h}` given :math:`x_t`, :math:`t` and :math:`h`."""
        raise NotImplementedError()

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Returns the final state (at ts[-1]) by updating initial state `x` through
        timesteps ts[0] to ts[-1].
        """
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_trajectories(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Returns the trajectory simulated by updating initial state `x` through
        timesteps ts[0] to ts[-1].
        """
        xs = [x.clone()]  # TODO: create a tensor and preallocate
        for t_idx in tqdm(range(len(ts) - 1)):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
