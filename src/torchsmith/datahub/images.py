import numpy as np
import torch


class DatasetImages(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray,
        *,
        mean: float,
        std: float,
    ) -> None:
        self.samples = torch.tensor((data - mean) / std).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
