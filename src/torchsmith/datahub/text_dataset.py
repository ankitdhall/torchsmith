import torch

from torchsmith.datahub.utils import chunk_batch
from torchsmith.tokenizers import TextTokenizer


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self, text: list[str], tokenizer: TextTokenizer, sequence_length: int = 128
    ) -> None:
        self.text = text
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        self.text_tokenized = list(self.tokenizer.encode_batch(iter(self.text)))
        assert all(
            len(text) == len(text_tok) - 2
            for text, text_tok in zip(self.text, self.text_tokenized)
        )
        self.samples = torch.tensor(
            chunk_batch(self.text_tokenized, sequence_length=sequence_length)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]
