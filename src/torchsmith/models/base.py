from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
import torch


class BaseModel(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        *,
        prefix: torch.Tensor,
        seq_len: int,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, np.ndarray]:
        raise NotImplementedError()
