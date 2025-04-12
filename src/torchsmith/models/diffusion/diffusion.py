import copy
from typing import Any

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from torchsmith.utils.pytorch import get_device

device = get_device()


class DiffusionModel(torch.nn.Module):
    def __init__(
        self, input_shape: int | tuple[int, ...], model: torch.nn.Module
    ) -> None:
        super().__init__()
        self.model = model.to(device)
        self.input_shape = input_shape
        self.num_input_dims = (
            input_shape if isinstance(input_shape, int) else int(np.prod(input_shape))
        )
        self.normal_distribution = MultivariateNormal(
            torch.zeros(self.num_input_dims), torch.eye(self.num_input_dims)
        )

    def sample_noise(self, num_samples: int) -> torch.Tensor:
        noise = self.normal_distribution.sample((num_samples,)).to(device=device)
        if isinstance(self.input_shape, tuple):
            noise = noise.reshape((num_samples, *self.input_shape))
        return noise

    @staticmethod
    def compute_noise_params(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = torch.cos(t * torch.pi / 2)
        sigma_t = torch.sin(t * torch.pi / 2)
        return alpha_t, sigma_t  # (B, 1), (B, 1)

    def add_noise(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, *_ = x.shape
        _size = (
            (batch_size, 1)  # (B, 1)
            if isinstance(self.input_shape, int)
            else (
                batch_size,
                *(1 for _ in range(len(self.input_shape))),
            )  # (B, 1, 1, 1)
        )
        t = torch.rand(size=_size, device=device)
        alpha_t, sigma_t = self.compute_noise_params(t)
        epsilon = self.sample_noise(batch_size)
        # print(
        #     f"_size:{_size}, t.shape:{t.shape}, alpha_t.shape:{alpha_t.shape}, "
        #     f"x.shape:{x.shape}, sigma_t.shape:{sigma_t.shape}, "
        #     f"epsilon.shape:{epsilon.shape}"
        # )
        x_t = alpha_t * x + sigma_t * epsilon
        return x_t, t, epsilon

    def forward(
        self, x: torch.Tensor, **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, t, epsilon = self.add_noise(x)
        epsilon_hat = self.model(x, t=t, **kwargs)
        return epsilon_hat, epsilon

    def ddpm_update(
        self,
        *,
        x: torch.Tensor,
        epsilon_hat: torch.Tensor,
        t: torch.Tensor,
        t_minus_1: torch.Tensor,
        clamp_to: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        batch_size, *_ = x.shape
        alpha_t, sigma_t = self.compute_noise_params(t)
        alpha_t_minus_1, sigma_t_minus_1 = self.compute_noise_params(t_minus_1)
        eta_t = (sigma_t_minus_1 / sigma_t) * (
            torch.sqrt(1 - (alpha_t**2 / alpha_t_minus_1**2))
        )

        x_hat = (x - sigma_t * epsilon_hat) / alpha_t
        if clamp_to is not None:
            x_hat = torch.clamp(x_hat, min=clamp_to[0], max=clamp_to[1])
        x = (
            alpha_t_minus_1 * x_hat
            + torch.sqrt(torch.clamp(sigma_t_minus_1**2 - eta_t**2, min=0))
            * epsilon_hat
            + (eta_t * self.sample_noise(batch_size))
        )
        return x

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        *,
        num_steps: int,
        clamp_to: tuple[float, float] | None = None,
        show_progress: bool = False,
        cfg_weight: float | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        self.model.eval()
        timesteps = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1)
        x = self.sample_noise(num_samples).to(device)
        steps = range(num_steps)
        if show_progress:
            steps = tqdm(steps)
        for step in steps:
            t = timesteps[step]
            t_minus_1 = timesteps[step + 1]

            if cfg_weight is None:
                epsilon_hat = self.model(
                    x,
                    t=t.repeat((num_samples, 1)).to(device=device, dtype=x.dtype),
                    **kwargs,
                )
            else:
                epsilon_hat_conditional = self.model(
                    x,
                    t=t.repeat((num_samples, 1)).to(device=device, dtype=x.dtype),
                    **kwargs,
                )
                kwargs_unconditional = copy.deepcopy(kwargs)
                kwargs_unconditional["y"] = torch.full_like(
                    kwargs_unconditional["y"],
                    fill_value=self.model.no_class_id,
                    device=device,
                )
                epsilon_hat_unconditional = self.model(
                    x,
                    t=t.repeat((num_samples, 1)).to(device=device, dtype=x.dtype),
                    **kwargs_unconditional,
                )
                epsilon_hat = epsilon_hat_unconditional + cfg_weight * (
                    epsilon_hat_conditional - epsilon_hat_unconditional
                )
            x = self.ddpm_update(
                x=x,
                epsilon_hat=epsilon_hat,
                t=t,
                t_minus_1=t_minus_1,
                clamp_to=clamp_to,
            )
        return x
