import dataclasses

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


Loss = VAELoss
