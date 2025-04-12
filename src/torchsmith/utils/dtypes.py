from typing import TYPE_CHECKING
from typing import Protocol
from typing import TypeVar
from typing import Union

import numpy as np
import torch

if TYPE_CHECKING:
    from torchsmith.models.gpt2 import GPT2Decoder
    from torchsmith.tokenizers import TextTokenizer
    from torchsmith.tokenizers.mnist_tokenizer import ColoredMNISTImageAndTextTokenizer
    from torchsmith.tokenizers.mnist_tokenizer import VQVAEColoredMNISTImageTokenizer

T = TypeVar("T")
X = TypeVar("X")
Y = TypeVar("Y")
ModuleT = TypeVar("ModuleT", bound="torch.nn.Module")
TokenType = Union[str, int]
TokenT = TypeVar("TokenT", str, int, str | int)


class GenerateSamplesProtocol(Protocol):
    def __call__(
        self,
        *,
        seq_len: int,
        tokenizer: Union[
            "TextTokenizer",
            "VQVAEColoredMNISTImageTokenizer",
            "ColoredMNISTImageAndTextTokenizer",
        ],
        transformer: "GPT2Decoder",
        decode: bool,
    ) -> torch.Tensor | tuple[list[np.ndarray], list[str]]: ...
