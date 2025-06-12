from abc import ABC
from abc import abstractmethod

import torch
from torch.func import jacrev
from torch.func import vmap


class Density(ABC):
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the score :math:`\\nabla log p_{density}(x)`."""
        x = x.unsqueeze(1)  # (batch_size, 1, ...)
        score = vmap(jacrev(self.log_density))(x)  # (batch_size, 1, 1, 1, ...)
        return score.squeeze((1, 2, 3))  # (batch_size, ...)


class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError()
