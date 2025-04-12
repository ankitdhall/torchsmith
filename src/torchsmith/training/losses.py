import torch
from einops import rearrange
from torch.nn import functional as F


def cross_entropy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # seq_len = (H W)
    y_hat = rearrange(y_hat, "B seq_len K -> B K seq_len")  # Predicted logits

    # print(f"Accuracy: {(y_hat.argmax(dim=1) == y).float().mean()}")

    # y: (B, H, W, C)
    return F.cross_entropy(y_hat, y.long())


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_hat, y)
