import dataclasses
from math import ceil
from typing import Any
from typing import cast

import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from torch.optim import Optimizer

from torchsmith.training.scheduler.base import BaseScheduler
from torchsmith.training.scheduler.base import BaseSchedulerConfig
from torchsmith.training.scheduler.base import SchedulerType


class CosineWarmupScheduler(BaseScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        *,
        warmup_steps: int,
        max_steps: int,  # TODO: rename max_steps to total_steps
    ) -> None:
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("`warmup_steps` must be a non-negative integer")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("`max_steps` must be a positive integer")
        if max_steps <= warmup_steps:
            raise ValueError("`max_steps` must be greater than `warmup_steps`")

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr_factor(self, epoch: int) -> float:
        # TODO: should this be called epoch or step?
        if epoch <= self.warmup_steps != 0:
            return 1.0 * epoch / self.warmup_steps
        return 0.5 * (
            1
            + np.cos(
                np.pi
                * (epoch - self.warmup_steps)
                / (self.max_steps - self.warmup_steps)
            )
        )

    def get_lr(self) -> list[float]:
        return [
            base_lr * self.get_lr_factor(self.last_epoch) for base_lr in self.base_lrs
        ]

    def visualize(self) -> None:
        x = np.arange(self.max_steps)
        y = [self.get_lr_factor(epoch) for epoch in x]
        plt.plot(x, y)
        plt.xlabel("Num steps")
        plt.ylabel("Learning rate factor")
        plt.title("Cosine warm-up scheduler")
        plt.show()


@dataclasses.dataclass(frozen=True)
class CosineWarmupSchedulerConfig(BaseSchedulerConfig):
    warmup_ratio: float | None = 0.5
    num_warmup_steps: int | None = None
    scheduler_type: SchedulerType = "cosine"  # TODO: Make this constant.

    def __post_init__(self):
        if self.warmup_ratio is None and self.num_warmup_steps is None:
            raise ValueError(
                "Either `warmup_ratio` or `num_warmup_steps` must be provided, "
                "currently both are None."
            )
        if self.warmup_ratio is not None and self.num_warmup_steps is not None:
            raise ValueError(
                "Either `warmup_ratio` or `num_warmup_steps` must be provided, "
                "currently both are provided."
            )
        if self.warmup_ratio and not (0 < self.warmup_ratio < 1):
            raise ValueError(
                f"Warmup ratio must be between 0 and 1 but was {self.warmup_ratio}."
            )

    def calculate_steps(
        self,
        *,
        num_epochs: int,
        num_batches_per_epoch: int | None = None,
        dataset_len: int | None = None,
        batch_size: int | None = None,
    ) -> tuple[int, int]:
        """Calculate total steps and warmup steps.

        Args:
            num_epochs: Number of training epochs.
            num_batches_per_epoch: Number of batches seen in a single training epoch.
            dataset_len: Length of the training dataset.
            batch_size: Batch size for training.

        Returns:
            2-tuple containing (total_steps, warmup_steps).
        """
        if num_batches_per_epoch is None:
            assert dataset_len is not None
            assert batch_size is not None
            num_batches_per_epoch = ceil(dataset_len / batch_size)

        total_steps = num_epochs * num_batches_per_epoch
        if self.warmup_ratio is not None:
            warmup_steps = int(total_steps * self.warmup_ratio)
        else:
            warmup_steps = cast(int, self.num_warmup_steps)

        _info = (
            f"{self.__class__.__name__}: calculating steps ...\n"
            f"Dataset length: {dataset_len} with batch size: {batch_size}\n"
            f"Number of batches seen per epoch: {num_batches_per_epoch}\n"
            f"Training for {num_epochs} epochs\n"
            f"Warming up for: {warmup_steps} out of {total_steps} total steps.\n"
        )
        print(_info)
        assert total_steps >= warmup_steps, (
            "Total steps must be greater than warmup steps."
        )
        return total_steps, warmup_steps

    def get_scheduler(
        self, optimizer: Optimizer, **kwargs: Any
    ) -> CosineWarmupScheduler:
        epochs = cast(int, kwargs.get("epochs"))
        num_batches_per_epoch = cast(int | None, kwargs.get("num_batches_per_epoch"))
        dataset_len = cast(int | None, kwargs.get("dataset_len"))
        batch_size = cast(int | None, kwargs.get("batch_size"))

        total_steps, warmup_steps = self.calculate_steps(
            num_epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            dataset_len=dataset_len,
            batch_size=batch_size,
        )
        return CosineWarmupScheduler(
            optimizer, warmup_steps=warmup_steps, max_steps=total_steps
        )
