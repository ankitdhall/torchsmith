import dataclasses
from typing import Union

import torch


@dataclasses.dataclass(frozen=True)
class VAELoss:
    KEYS = [  # noqa: RUF012
        "negative_ELBO",
        "reconstruction_loss",
        "KL_div_loss",
    ]
    DESCRIPTION = "-ELBO, recon. loss, KL-div"

    negative_ELBO: torch.Tensor
    reconstruction_loss: torch.Tensor
    KL_div_loss: torch.Tensor

    def get_total_loss(self) -> torch.Tensor:
        return self.negative_ELBO


@dataclasses.dataclass(frozen=True)
class VQVAELoss:
    KEYS = [  # noqa: RUF012
        "total_loss",
        "reconstruction_loss",
        "codebook_encoder_loss",
    ]
    DESCRIPTION = "total loss, recon. loss, codebook<->encoder loss"
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    codebook_encoder_loss: torch.Tensor

    def get_total_loss(self) -> torch.Tensor:
        return self.total_loss


Loss = Union[VAELoss, VQVAELoss]
