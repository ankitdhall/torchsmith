import torch
from einops import rearrange

from torchsmith.models.gpt2.modules import ScaledDotProductAttention
from torchsmith.utils.pytorch import add_save_load
from torchsmith.utils.pytorch import get_device

device = get_device()


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
    return embedding.to(device)


class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, dim_model: int, grid_size: int, max_len: int = 10000) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.max_len = max_len
        self.grid_size = grid_size
        position_w = torch.arange(0, grid_size).float()  # (grid_size,)
        position_h = torch.arange(0, grid_size).float()  # (grid_size,)
        grid = torch.meshgrid(position_w, position_h)  # Row-major ordering.
        grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)
        encoding_w = self.get_1d_encoding(
            num_dims=dim_model // 2, positions=grid[0].reshape(-1)
        )
        encoding_h = self.get_1d_encoding(
            num_dims=dim_model // 2, positions=grid[1].reshape(-1)
        )
        self.register_buffer(
            "encoding", torch.cat([encoding_w, encoding_h], dim=-1).to(device)
        )

    def get_1d_encoding(self, num_dims: int, positions: torch.Tensor) -> torch.Tensor:
        assert len(positions.shape) == 1
        positions = positions.reshape((-1, 1))  # positions: (num_positions, 1)
        half_dim = num_dims // 2
        i = torch.arange(0, half_dim).float()
        denominator = torch.pow(self.max_len, (i / half_dim))  # (dim_model/2,)
        encoding_sin = torch.sin(
            positions / denominator
        )  # (num_positions, dim_model/2)
        encoding_cos = torch.cos(
            positions / denominator
        )  # (num_positions, dim_model/2)
        encoding = torch.cat(
            [encoding_sin, encoding_cos], dim=-1
        )  # (num_positions, dim_model)
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid) -> torch.Tensor:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=-1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def modulate(
    x: torch.Tensor, *, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    # x: (B, L, D) shift: (B, D) scale: (B, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PointWiseFeedForwardWithSiLU(torch.nn.Module):
    def __init__(self, *, dim_model: int, dim_feed_forward: int) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(dim_model, dim_feed_forward)
        self.activation = torch.nn.SiLU()
        self.linear_2 = torch.nn.Linear(dim_feed_forward, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class MultiHeadSelfAttentionWithoutBias(torch.nn.Module):
    def __init__(
        self, *, num_heads: int, dim_embeddings: int, auto_regressive_mask: bool
    ) -> None:
        super().__init__()
        self.dim_embeddings = dim_embeddings
        self.d_k = dim_embeddings // num_heads
        self.d_v = dim_embeddings // num_heads
        self.num_heads = num_heads

        # Linear projection as a single linear layer for efficiency.
        self.project_q = torch.nn.Linear(dim_embeddings, dim_embeddings, bias=False)
        self.project_kv = torch.nn.Linear(
            dim_embeddings, 2 * dim_embeddings, bias=False
        )
        self.attention = ScaledDotProductAttention(auto_regressive_mask)

        self.project_output = torch.nn.Linear(
            dim_embeddings, dim_embeddings, bias=False
        )

    def forward(
        self, x: torch.Tensor, use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.project_kv(x).split(self.dim_embeddings, dim=-1)

        q = self.project_q(x)
        split_pattern = "batch seq_len (num_heads d_k) -> batch num_heads seq_len d_k"
        q = rearrange(q, split_pattern, num_heads=self.num_heads)
        k = rearrange(k, split_pattern, num_heads=self.num_heads)
        v = rearrange(v, split_pattern, num_heads=self.num_heads)
        output, attention = self.attention(q, k, v)
        concat_pattern = "batch num_heads seq_len d_k -> batch seq_len (num_heads d_k)"
        output = rearrange(output, concat_pattern)
        output = self.project_output(output)
        return output, attention


class DiTBlock(torch.nn.Module):
    def __init__(self, *, dim_model: int, num_heads: int) -> None:
        super().__init__()
        self.c_transform = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(dim_model, dim_model * 6)
        )

        self.ln1 = torch.nn.LayerNorm(dim_model, elementwise_affine=False)
        self.ln2 = torch.nn.LayerNorm(dim_model, elementwise_affine=False)
        self.attention = MultiHeadSelfAttentionWithoutBias(
            num_heads=num_heads,
            dim_embeddings=dim_model,
            auto_regressive_mask=True,
        )
        self.ffn = self.feed_forward_network = PointWiseFeedForwardWithSiLU(
            dim_model=dim_model, dim_feed_forward=dim_model * 4
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        # c: (B, D)
        c = self.c_transform(c)  # (B, D) -> (B, D*6)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(
            6, dim=-1
        )

        h = self.ln1(x)
        h = modulate(h, shift=shift_msa, scale=scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attention(h)[0]

        h = self.ln2(x)
        h = modulate(h, shift=shift_mlp, scale=scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(h)

        return x


class FinalLayer(torch.nn.Module):
    def __init__(
        self, *, dim_model: int, patch_size: int, num_channels_out: int
    ) -> None:
        super().__init__()
        self.c_transform = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(dim_model, dim_model * 2)
        )

        self.ln = torch.nn.LayerNorm(dim_model, elementwise_affine=False)
        self.linear = torch.nn.Linear(
            dim_model, patch_size * patch_size * num_channels_out
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        # c: (B, D)
        c = self.c_transform(c)  # (B, D*2)
        shift, scale = c.chunk(2, dim=1)

        x = self.ln(x)
        x = modulate(x, shift=shift, scale=scale)
        x = self.linear(x)
        return x


@add_save_load
class DiT(torch.nn.Module):
    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        patch_size: int,
        dim_model: int,
        num_heads: int,
        num_blocks: int,
        num_classes: int,
        cfg_dropout_prob: float,
        time_embedding_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.patch_size = patch_size
        self.input_channels = input_shape[0]
        self.cfg_dropout_prob = cfg_dropout_prob
        self.time_embedding_size = time_embedding_size
        self.no_class_id = num_classes
        self.patch_embedder = torch.nn.Conv2d(
            self.input_channels,
            self.dim_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        assert input_shape[1] == input_shape[2]
        self.positional_embeddings = PositionalEncoding2D(
            dim_model=dim_model, grid_size=input_shape[1] // patch_size
        )
        self.blocks = torch.nn.ModuleList(
            [
                DiTBlock(dim_model=dim_model, num_heads=num_heads)
                for _ in range(num_blocks)
            ]
        )
        self.embeddings = torch.nn.Embedding(num_classes + 1, dim_model)

        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.time_embedding_size, self.dim_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_model, self.dim_model),
        )

        self.final_layer = FinalLayer(
            dim_model=dim_model,
            patch_size=patch_size,
            num_channels_out=self.input_channels,
        )
        self.fold = torch.nn.Fold(
            output_size=(input_shape[1], input_shape[2]),
            kernel_size=(self.patch_size, self.patch_size),
            stride=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        # Given x (B, C, H, W) - image, y (B) - class label, t (B) - diffusion timestep
        # x: (B, L, D)
        # c: (B, D)
        # B x C x H x W -> B x (H // P * W // P) x D, P is patch_size
        t = t.squeeze()  # TODO: DiffusionModel.sample passes tensor with shape (B, 1)!
        assert x.ndim == 4
        assert t.ndim == 1
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0] == t.shape[0]
        x = self.patch_embedder(
            x
        )  # (B, C, H, W) -> (B, D, H // patch_size, W // patch_size)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.dim_model, -1)  # (B, D, num_patches)
        x = x.permute(0, 2, 1)  # (B, D, num_patches) -> (B, num_patches, D)
        x = self.positional_embeddings(x)

        t = self.time_embedding(timestep_embeddings(t, self.time_embedding_size))
        if self.training:
            # Randomly replace the class label with <no_class> ID
            # to train unconditional image generation
            mask = torch.rand(size=y.shape, device=device) < self.cfg_dropout_prob
            y = y.masked_fill(mask, self.no_class_id)
        y = self.embeddings(y)
        conditioning = t + y

        for block in self.blocks:
            x = block(x, conditioning)

        x = self.final_layer(
            x, conditioning
        )  # (B, num_patches, D) -> (B, num_patches, P*P*C)
        x = x.permute(0, 2, 1)  # (B, P*P*C, num_patches)
        x = self.fold(x)  # (B, P*P*C, num_patches) -> (B, C, H, W)
        return x
