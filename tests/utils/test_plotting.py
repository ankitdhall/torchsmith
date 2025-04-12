from pathlib import Path

import numpy as np
import pytest

from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.plotting import suppress_plot


def _get_values() -> tuple[list[float], list[float]]:
    total_epochs = 5
    num_batches_per_epoch = 40
    rng = np.random.default_rng(0)
    train_losses = rng.random(total_epochs * num_batches_per_epoch)
    test_losses = np.linspace(1.0, 0.05, total_epochs + 1)
    return train_losses.tolist(), test_losses.tolist()


@pytest.mark.parametrize(
    ["train_losses", "test_losses"],
    [
        (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # 1st-----, 2nd-----, 3rd-----, 4th-----, 5th-----
            [1.0, 0.5, 0.3, 0.15, 0.1, 0.05],
            # 0th, 1st, 2nd, 3rd, 4th, 5th
        ),
        (_get_values()),
    ],
)
@pytest.mark.parametrize("save", [True, False])
def test_plot_losses(
    train_losses: list[float], test_losses: list[float], save: bool, tmp_path: Path
) -> None:
    with suppress_plot():
        plot_losses(
            train_losses,
            test_losses=test_losses,
            save_dir=tmp_path if save else None,
            show=True,
        )
