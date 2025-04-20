import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def plot_samples(
    images: np.ndarray,
    *,
    num_rows: int = 10,
    save_dir: Path | None = None,
    show: bool = False,
    filename: str | None = None,
):
    if images.ndim != 4:
        raise ValueError(
            f"`images` must be of shape (B, C, H, W) but found: {images.shape}"
        )
    if images.shape[1] not in [1, 3]:
        raise ValueError(
            f"`images` must be of shape (B, C, H, W) with 1 or 3 channels "
            f"but found: {images.shape}"
        )
    if np.max(images) > 255 or np.min(images) < 0:
        warnings.warn(
            "Image pixels should be between [0, 255]. But found (min, max) to be "
            f"({np.min(images)}, {np.max(images)}).",
            stacklevel=2,
        )

    images = torch.FloatTensor(images) / 255
    grid_img = make_grid(images, nrow=images.shape[0] // num_rows)
    plt.figure()
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_dir / f"{filename}.png" if filename is not None else "samples.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()
