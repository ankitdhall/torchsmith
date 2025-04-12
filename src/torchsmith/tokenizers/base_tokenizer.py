from abc import ABC
from abc import abstractmethod
from typing import Generic

from torchsmith.utils.dtypes import TokenT
from torchsmith.utils.pytorch import get_device

device = get_device()


class BaseTokenizer(ABC, Generic[TokenT]):
    def __init__(
        self,
        *,
        n_jobs: int = 1,
        batch_size: int = 1000,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self._n_jobs = n_jobs
        self.batch_size = batch_size

    @property
    @abstractmethod
    def token_to_id(self) -> dict[TokenT, int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def id_to_token(self) -> dict[int, TokenT]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.token_to_id)

    def to_token(self, id: int) -> TokenT:
        return self.id_to_token[id]

    def to_id(self, token: TokenT) -> int:
        return self.token_to_id[token]

    @property
    def tokens(self) -> set[TokenT]:
        return set(self.token_to_id.keys())

    @property
    def n_jobs(self) -> int:
        return self._n_jobs
