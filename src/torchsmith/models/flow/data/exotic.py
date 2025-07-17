import numpy as np
import torch
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

from torchsmith.models.flow.data import Sampleable
from torchsmith.utils.pytorch import get_device

device = get_device()


class MoonsSampleable(Sampleable):
    def __init__(
        self,
        noise_std: float = 0.05,
        scale: float = 5.0,
        offset: torch.Tensor | None = None,
    ) -> None:
        self.noise_std = noise_std
        self.scale = scale
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def num_dims(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        samples, _ = make_moons(
            n_samples=num_samples,
            noise=self.noise_std,
            random_state=None,  # Allow for random generation each time
        )
        return (
            self.scale * torch.from_numpy(samples.astype(np.float32)).to(device)
            + self.offset
        )


class CirclesSampleable(Sampleable):
    def __init__(
        self, noise_std: float = 0.05, scale=5.0, offset: torch.Tensor | None = None
    ) -> None:
        self.noise_std = noise_std
        self.scale = scale
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def num_dims(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        samples, _ = make_circles(
            n_samples=num_samples, noise=self.noise_std, factor=0.5, random_state=None
        )
        return (
            self.scale * torch.from_numpy(samples.astype(np.float32)).to(device)
            + self.offset
        )


class CheckerboardSampleable(Sampleable):
    def __init__(self, grid_size: int = 3, scale=5.0) -> None:
        self.grid_size = grid_size
        self.scale = scale

    @property
    def num_dims(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0, 2).to(device)
        while samples.shape[0] < num_samples:
            new_samples = (torch.rand(num_samples, 2).to(device) - 0.5) * 2 * self.scale
            x_mask = (
                torch.floor((new_samples[:, 0] + self.scale) / grid_length) % 2 == 0
            )
            y_mask = (
                torch.floor((new_samples[:, 1] + self.scale) / grid_length) % 2 == 0
            )
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)
        return samples[:num_samples]
