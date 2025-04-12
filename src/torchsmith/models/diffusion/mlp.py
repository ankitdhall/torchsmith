import numpy as np
import torch

from torchsmith.utils.pytorch import add_save_load


@add_save_load
class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int,
        num_hidden_layers: int = 4,
        hidden_layer_dim: int = 64,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_layer_dim, hidden_layer_dim), torch.nn.ReLU()
                )
                for _ in range(num_hidden_layers)
            ],
            torch.nn.Linear(hidden_layer_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, t], dim=1)
        x = self.layers(x)
        return x


class Dataset2D(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.samples = torch.tensor((data - mean) / std).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
