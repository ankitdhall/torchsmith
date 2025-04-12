import math
from typing import Literal
from typing import overload

import torch
from einops import rearrange
from torch import nn
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn.functional import softmax

from torchsmith.utils.pytorch import get_device

device = get_device()


class ImageGPTLayerNorm(nn.Module):
    def __init__(self, hidden_size: tuple[int], eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, tensor: torch.Tensor) -> tuple:
        # input is not mean centered
        return (
            tensor
            / torch.sqrt(
                torch.mean(torch.square(tensor), axis=-1, keepdim=True) + self.eps
            )
            * self.weight.data[..., :]
        )


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, *, dim_model: int, dim_feed_forward: int) -> None:
        super().__init__()
        self.linear_1 = Linear(dim_model, dim_feed_forward)
        self.activation = torch.nn.GELU()
        self.linear_2 = Linear(dim_feed_forward, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, auto_regressive_mask: bool) -> None:
        super().__init__()
        self.auto_regressive_mask = auto_regressive_mask

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, d_k = Q.size()

        # (batch_size, num_heads, seq_len, seq_len)
        scaled_dot_product = Q @ K.transpose(2, 3) / math.sqrt(d_k)

        if self.auto_regressive_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(Q.device)
            scaled_dot_product = scaled_dot_product.masked_fill(
                mask == 0, float("-inf")
            )

        # Each row sums to 1.
        # (batch_size, num_heads, seq_len, seq_len)
        weights = softmax(scaled_dot_product, dim=-1)

        # Each row vector is a weighted sum of the vectors in V.
        # (batch_size, num_heads, seq_len, d_v)
        # (batch_size, num_heads, seq_len, seq_len)
        return weights @ V, weights


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self, *, num_heads: int, dim_embeddings: int, auto_regressive_mask: bool
    ) -> None:
        super().__init__()
        self.dim_embeddings = dim_embeddings
        self.d_k = dim_embeddings // num_heads
        self.d_v = dim_embeddings // num_heads
        self.num_heads = num_heads

        # Linear projection as a single linear layer for efficiency.
        # Bias is still interacting between different heads.
        self.project_q = Linear(dim_embeddings, dim_embeddings)
        self.project_kv = Linear(dim_embeddings, 2 * dim_embeddings)
        self.attention = ScaledDotProductAttention(auto_regressive_mask)

        self.project_output = Linear(dim_embeddings, dim_embeddings)

        self.kv_cache: dict[str, torch.Tensor] = {}

    def clear_cache(self, batch_size: int, seq_len: int) -> None:
        #         self.kv_cache = {}
        self.kv_cache = {
            "K": torch.empty((batch_size, seq_len, self.dim_embeddings), device=device),
            "V": torch.empty((batch_size, seq_len, self.dim_embeddings), device=device),
        }

    def forward(
        self, x: torch.Tensor, use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, seq_len, dim_embeddings) * (dim_embeddings, dim_embeddings)
        if use_cache:
            # print(f"In forward: using cache; x: {x.shape}")
            assert self.kv_cache != {}
            current_seq_len = x.shape[1]

            # print(f"In cache: K {self.kv_cache['K'].shape} V
            # {self.kv_cache['V'].shape}")
            k_current, v_current = self.project_kv(x[:, -1, :]).split(
                self.dim_embeddings, dim=-1
            )
            self.kv_cache["K"][:, current_seq_len - 1, :] = k_current
            self.kv_cache["V"][:, current_seq_len - 1, :] = v_current

            k = self.kv_cache["K"][:, :current_seq_len, :]
            v = self.kv_cache["V"][:, :current_seq_len, :]

            # print(f"Combining: K {k.shape} V {v.shape}")
        else:
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


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim_model: int, device: str, max_len: int = 5000) -> None:
        super().__init__()
        self.encoding = torch.zeros(max_len, dim_model, device=device)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        i = torch.arange(0, dim_model // 2).float()
        denominator = torch.pow(10000, (2 * i / dim_model))
        # (max_len, 1) / (dim_model,) -> (max_len, dim_model)
        self.encoding[:, 0::2] = torch.sin(position / denominator)
        self.encoding[:, 1::2] = torch.cos(position / denominator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[: x.size(1), :]


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        dim_model: int,
        dim_feed_forward: int,
        auto_regressive_mask: bool,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            dim_embeddings=dim_model,
            auto_regressive_mask=auto_regressive_mask,
        )
        self.norm_attention = LayerNorm(dim_model)
        self.dropout_attention = Dropout(p=dropout)
        self.feed_forward_network = PointWiseFeedForward(
            dim_model=dim_model, dim_feed_forward=dim_feed_forward
        )
        self.norm_ffn = LayerNorm(dim_model)
        self.dropout_ffn = Dropout(p=dropout)

    def clear_cache(self, batch_size: int, seq_len: int) -> None:
        self.attention.clear_cache(batch_size, seq_len)

    @overload
    def forward(
        self, x: torch.Tensor, return_attention: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def forward(
        self, x: torch.Tensor, return_attention: Literal[False]
    ) -> torch.Tensor: ...

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # Sub-layer 1: Multi-head self-attention mechanism.
        attention_output, attention = self.attention(x)
        x_out_sa = self.norm_attention(self.dropout_attention(attention_output) + x)

        # Sub-layer 2: Feed-forward network.
        x_out_ffn = self.norm_ffn(
            self.dropout_ffn(self.feed_forward_network(x_out_sa)) + x_out_sa
        )
        if return_attention:
            return x_out_ffn, attention
        else:
            return x_out_ffn


class GPT2Layer(torch.nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        dim_model: int,
        dim_feed_forward: int,
        auto_regressive_mask: bool,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            num_heads=num_heads,
            dim_embeddings=dim_model,
            auto_regressive_mask=auto_regressive_mask,
        )
        # self.attention = nn.MultiheadAttention(dim_model, num_heads)
        self.norm_attention = LayerNorm(dim_model)
        self.dropout_attn = Dropout(p=dropout)

        self.feed_forward_network = PointWiseFeedForward(
            dim_model=dim_model, dim_feed_forward=dim_feed_forward
        )
        self.norm_ffn = LayerNorm(dim_model)
        self.dropout_ffn = Dropout(p=dropout)

    def clear_cache(self, batch_size: int, seq_len: int) -> None:
        self.attention.clear_cache(batch_size, seq_len)

    @overload
    def forward(
        self, x: torch.Tensor, return_attention: Literal[True], use_cache: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        return_attention: Literal[True],
        use_cache: Literal[False],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        return_attention: Literal[False],
        use_cache: Literal[True],
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        return_attention: Literal[False],
        use_cache: Literal[False],
    ) -> torch.Tensor: ...

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # Sub-layer 1: Multi-head self-attention mechanism.
        x_normalized = self.norm_attention(self.dropout_attn(x))
        attention_output, attention = self.attention(x_normalized, use_cache=use_cache)

        # attn_mask = torch.full(
        #     (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        # )
        # attn_mask = torch.triu(attn_mask, diagonal=1)
        # attention_output, attention = self.attention(
        # x_normalized, x_normalized, x_normalized, attn_mask=attn_mask)

        x_out_sa = attention_output + x

        # Sub-layer 2: Feed-forward network.
        x_out_ffn = self.feed_forward_network(self.norm_ffn(self.dropout_ffn(x_out_sa)))
        x_out_ffn = x_out_ffn + x_out_sa

        if return_attention:
            return x_out_ffn, attention
        else:
            return x_out_ffn
