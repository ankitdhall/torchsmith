import dataclasses

from torchsmith.training.optimizer import AdamConfig
from torchsmith.training.optimizer import OptimizerConfig
from torchsmith.training.scheduler import CosineWarmupSchedulerConfig
from torchsmith.training.scheduler import SchedulerConfig


@dataclasses.dataclass(frozen=True)
class GPT2Config:
    seq_len: int
    dim_model: int = 128
    num_heads: int = 4
    dim_feed_forward: int = dim_model * 4
    num_stack: int = 2
    dropout: float = 0.0


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 512
    num_epochs: int = 100
    num_workers: int = 2
    optimizer_config: OptimizerConfig = dataclasses.field(default_factory=AdamConfig)
    scheduler_config: SchedulerConfig | None = dataclasses.field(
        default_factory=CosineWarmupSchedulerConfig
    )
    # TODO: include scheduler and optimizer creation as part of this class?
