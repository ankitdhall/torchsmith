import numpy as np
import torch

from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.utils.pytorch import get_device

device = get_device()


class ImagesWithVQVAEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: np.ndarray,
        tokenizer: VQVAEImageTokenizer,
    ) -> None:
        self.tokenizer = tokenizer
        self.samples = torch.tensor(
            self.tokenizer.encode(images, drop_bos=False, drop_eos=True)
        )
        self.sequence_length = self.samples.shape[-1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
