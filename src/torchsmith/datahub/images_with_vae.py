from typing import Any

import numpy as np
import torch

from torchsmith.models.external._vae import VAE
from torchsmith.utils.pyutils import batched


class DatasetImagesWithVAE(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray,
        *,
        labels: np.ndarray,
        scale_factor: float,
        vae: VAE,
        mean: float = 0.5,
        std: float = 0.5,
        batch_size: int = 1000,
    ) -> None:
        data = (data - mean) / std
        samples_list = []
        for batch in batched(data, batch_size):
            samples_list.append(vae.encode(np.array(batch)).cpu())
        self.samples = torch.cat(samples_list, dim=0) / scale_factor
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.samples[idx], {"y": self.labels[idx]}
