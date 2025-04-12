import dataclasses
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Literal

from torch import optim
from torch.optim import Optimizer

SchedulerType = Literal["cosine"]


class BaseScheduler(ABC, optim.lr_scheduler.LRScheduler):
    @abstractmethod
    def visualize(self) -> None:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class BaseSchedulerConfig:
    scheduler_type: SchedulerType

    def get_scheduler(self, optimizer: Optimizer, **kwargs: Any) -> BaseScheduler:
        raise NotImplementedError


def get_scheduler(
    config: BaseSchedulerConfig, *, optimizer: Optimizer, **kwargs: Any
) -> BaseScheduler:
    return config.get_scheduler(optimizer, **kwargs)
