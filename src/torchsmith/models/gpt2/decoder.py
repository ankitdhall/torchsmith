import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import Linear
from tqdm import tqdm

from torchsmith.models.base import BaseModel
from torchsmith.models.gpt2.modules import GPT2Layer
from torchsmith.training.config import GPT2Config
from torchsmith.utils.pytorch import add_save_load
from torchsmith.utils.pytorch import get_device

device = get_device()


@add_save_load
class GPT2Decoder(BaseModel):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        seq_len: int,
        dim_model: int,
        num_heads: int,
        dim_feed_forward: int,
        num_stack: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.seq_len = seq_len

        self.embeddings = torch.nn.Embedding(self.vocabulary_size, dim_model)

        self.positional_embeddings = torch.nn.Embedding(seq_len, dim_model)
        self.layers = torch.nn.ModuleList(
            GPT2Layer(
                num_heads=num_heads,
                dim_model=dim_model,
                dim_feed_forward=dim_feed_forward,
                dropout=dropout,
                auto_regressive_mask=True,
            )
            for _ in range(num_stack)
        )
        self.dropout = Dropout(p=dropout)
        self.norm = LayerNorm(dim_model)
        self.linear = Linear(dim_model, self.vocabulary_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("feed_forward_network.linear_2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_stack))

    @classmethod
    def from_config(cls, vocab_size: int, *, config: GPT2Config) -> "GPT2Decoder":
        return GPT2Decoder(
            vocabulary_size=vocab_size,
            seq_len=config.seq_len,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            dim_feed_forward=config.dim_feed_forward,
            num_stack=config.num_stack,
            dropout=config.dropout,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, *, use_cache: bool = False) -> torch.Tensor:  # type: ignore
        batch_size, seq_len = x.shape

        # (B, (1 + H*W), dim)
        x = self.embeddings(x) + self.positional_embeddings(
            torch.arange(seq_len, device=x.device)
        )
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, use_cache=use_cache)
        x = self.linear(self.norm(x))  # (B, (1 + H*W), K)
        return x

    def clear_cache(self, batch_size: int, seq_len: int) -> None:
        for layer in self.layers:
            layer.clear_cache(batch_size, seq_len)

    def sample(  # type: ignore
        self,
        num_samples: int,
        *,
        prefix: torch.Tensor,
        seq_len: int,  # TODO: remove if unneeded
        use_cache: bool = False,
        exclude_indices: set[int] | None = None,
        include_indices: set[int] | None = None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        if exclude_indices and include_indices:
            raise ValueError(
                "Both `exclude_indices` and `include_indices` cannot be "
                "set simultaneously."
            )

        if exclude_indices is not None:
            mask_logits_to_neg_inf = torch.full(
                (1, self.vocabulary_size), False, device=device
            )
            mask_logits_to_neg_inf[:, list(exclude_indices)] = True
        if include_indices is not None:
            mask_logits_to_neg_inf = torch.full(
                (1, self.vocabulary_size), True, device=device
            )
            mask_logits_to_neg_inf[:, list(include_indices)] = False

        self.eval()
        prefix = prefix.to(device)

        samples = torch.zeros((num_samples, seq_len), device=device, dtype=int)
        print("Sampling ...")

        # TODO: make batch size sep from num_samples
        batch_size = num_samples
        num_prefixes, prefix_len = prefix.shape
        assert num_prefixes == num_samples

        samples[:, :prefix_len] = prefix

        inference_times = np.zeros(seq_len)
        with torch.no_grad():
            self.clear_cache(batch_size, seq_len=seq_len)
            for i in tqdm(range(prefix_len, seq_len), desc="Sampling tokens ..."):
                start = time.time()
                # TODO: check the caching works as expected
                logits = self.forward(samples, use_cache=use_cache)
                inference_times[i] = time.time() - start
                logits = logits[:, i - 1, :]  # (B, i, K) -> (B, K)
                if exclude_indices or include_indices:
                    logits = logits.masked_fill(
                        mask_logits_to_neg_inf, -torch.inf
                    )  # (B, K) * (1, K) -> (B, K)
                prob = torch.softmax(logits, dim=-1)  # (B, K)
                generated = torch.multinomial(prob, 1)  # (B, K) -> (B, 1)
                samples[:, i] = generated.squeeze(-1)  # (B, 1) -> (B,)

            return samples, inference_times
