import dataclasses


@dataclasses.dataclass(frozen=True)
class AdamConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0


OptimizerConfig = AdamConfig
