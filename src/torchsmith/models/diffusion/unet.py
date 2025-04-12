import torch
import torch.nn.functional as F

from torchsmith.utils.pytorch import add_save_load
from torchsmith.utils.pytorch import get_device

device = get_device()


class Downsample(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Upsample(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.layer(x)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self, *, num_channels_in: int, num_channels_out: int, num_t_emb_channels: int
    ) -> None:
        super().__init__()
        self.sub_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                num_channels_in, num_channels_out, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.GroupNorm(num_groups=8, num_channels=num_channels_out),
            torch.nn.SiLU(),
        )

        self.fc_t_emb = torch.nn.Linear(num_t_emb_channels, num_channels_out)

        self.sub_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                num_channels_out, num_channels_out, kernel_size=3, padding=1
            ),
            torch.nn.GroupNorm(num_groups=8, num_channels=num_channels_out),
            torch.nn.SiLU(),
        )

        self.conv_shortcut = None
        if num_channels_in != num_channels_out:
            self.conv_shortcut = torch.nn.Conv2d(
                num_channels_in, num_channels_out, kernel_size=1, stride=1
            )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.sub_block_1(x)
        t_emb_h = self.fc_t_emb(t_emb)
        h += t_emb_h.unsqueeze(-1).unsqueeze(-1)
        h = self.sub_block_2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


def timestep_embeddings(
    timesteps: torch.Tensor, num_dim: int, max_period: int = 10000
) -> torch.Tensor:
    half_dim = num_dim // 2
    frequency = torch.exp(
        -torch.log(torch.tensor([max_period])) * torch.arange(0, half_dim) / half_dim
    ).to(device=device)
    args = timesteps[:, None] * frequency[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).squeeze()
    if num_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@add_save_load
class UNet(torch.nn.Module):
    def __init__(
        self,
        *,
        num_channels_in: int,
        num_hidden_dims: list[int],
        num_blocks_per_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.first_hidden_dim = num_hidden_dims[0]
        self.num_hidden_dims = num_hidden_dims
        self.num_blocks_per_hidden_dim = num_blocks_per_hidden_dim
        self.num_t_emb_channels = self.first_hidden_dim * 4

        self.t_embeddings_fc = torch.nn.Sequential(
            torch.nn.Linear(self.first_hidden_dim, self.num_t_emb_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.num_t_emb_channels, self.num_t_emb_channels),
        )

        self.initial_conv = torch.nn.Conv2d(
            num_channels_in, self.first_hidden_dim, 3, padding=1
        )

        down_blocks: list[torch.nn.Module] = []
        prev_num_channels = self.first_hidden_dim
        down_block_num_channels = [prev_num_channels]
        for hidden_dim_index, hidden_dim in enumerate(self.num_hidden_dims):
            for _ in range(self.num_blocks_per_hidden_dim):
                r = ResidualBlock(
                    num_channels_in=prev_num_channels,
                    num_channels_out=hidden_dim,
                    num_t_emb_channels=self.num_t_emb_channels,
                )
                down_blocks.append(r)
                prev_num_channels = hidden_dim
                down_block_num_channels.append(prev_num_channels)
            if hidden_dim_index != len(self.num_hidden_dims) - 1:
                r = Downsample(prev_num_channels)
                down_blocks.append(r)
                down_block_num_channels.append(prev_num_channels)
        self.down_blocks = torch.nn.ModuleList(down_blocks)

        self.mid_blocks = torch.nn.ModuleList(
            [
                ResidualBlock(
                    num_channels_in=prev_num_channels,
                    num_channels_out=prev_num_channels,
                    num_t_emb_channels=self.num_t_emb_channels,
                ),
                ResidualBlock(
                    num_channels_in=prev_num_channels,
                    num_channels_out=prev_num_channels,
                    num_t_emb_channels=self.num_t_emb_channels,
                ),
            ]
        )

        up_blocks: list[torch.nn.Module] = []
        for hidden_dim_index, hidden_dim in list(enumerate(self.num_hidden_dims))[::-1]:
            for block_index in range(self.num_blocks_per_hidden_dim + 1):
                down_block_num_channel = down_block_num_channels.pop()
                r = ResidualBlock(
                    num_channels_in=prev_num_channels + down_block_num_channel,
                    num_channels_out=hidden_dim,
                    num_t_emb_channels=self.num_t_emb_channels,
                )
                up_blocks.append(r)
                prev_num_channels = hidden_dim
                if hidden_dim_index and block_index == self.num_blocks_per_hidden_dim:
                    r = Upsample(prev_num_channels)
                    up_blocks.append(r)
        self.up_blocks = torch.nn.ModuleList(up_blocks)

        self.final_block = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=8, num_channels=prev_num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(prev_num_channels, num_channels_in, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embeddings = timestep_embeddings(t, self.first_hidden_dim)
        t_embeddings = self.t_embeddings_fc(t_embeddings)

        h = self.initial_conv(x)

        intermediate_activations_down = [h]
        index_down_block = 0
        for hidden_dim_index, _ in enumerate(self.num_hidden_dims):
            for _ in range(self.num_blocks_per_hidden_dim):
                h = self.down_blocks[index_down_block](h, t_embeddings)
                index_down_block += 1
                intermediate_activations_down.append(h)
            if hidden_dim_index != len(self.num_hidden_dims) - 1:
                h = self.down_blocks[index_down_block](h)
                index_down_block += 1
                intermediate_activations_down.append(h)

        h = self.mid_blocks[0](h, t_embeddings)
        h = self.mid_blocks[1](h, t_embeddings)

        index_up_block = 0
        for hidden_dim_index, _ in list(enumerate(self.num_hidden_dims))[::-1]:
            for block_index in range(self.num_blocks_per_hidden_dim + 1):
                intermediate_activation = intermediate_activations_down.pop()
                h_with_shortcut = torch.cat([h, intermediate_activation], dim=1)
                h = self.up_blocks[index_up_block](h_with_shortcut, t_embeddings)
                index_up_block += 1
                if hidden_dim_index and block_index == self.num_blocks_per_hidden_dim:
                    h = self.up_blocks[index_up_block](h)
                    index_up_block += 1

        return self.final_block(h)
