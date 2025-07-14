import torch


class MLPForMatching(torch.nn.Module):
    def __init__(self, num_dims: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers = []
        input_dim = num_dims + 1  # +1 for the time dimension
        for output_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, output_dim))
            layers.append(torch.nn.SiLU())
            input_dim = output_dim
        layers.append(torch.nn.Linear(input_dim, num_dims))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, t], dim=-1))
