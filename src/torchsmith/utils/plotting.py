from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


@contextmanager
def suppress_plot() -> Iterator[None]:
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = original_show


def plot_histogram(
    array: list | np.ndarray | pd.Series, *, variable_name: str, num_bins: int = 50
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(array, bins=num_bins, edgecolor="k")
    plt.title(f"Distribution of {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("Frequency")
    plt.show()


def plot_images(
    images: list[np.ndarray],
    *,
    titles: list[str] | None = None,
    main_title: str | None = None,
    max_cols: int = 3,
):
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return

    # Determine grid size
    cols = min(max_cols, num_images)  # Limit number of columns
    rows = (num_images + cols - 1) // cols  # Compute required rows

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Set main title if provided
    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight="bold")

    # Ensure axes is iterable (handles both single and multiple rows)
    axes = np.array(axes).reshape(-1)

    # Plot each image
    for i in range(num_images):
        decoded_image = images[i].squeeze()
        decoded_image = np.floor(decoded_image.astype("float32") / 3 * 255).astype(int)
        axes[i].imshow(decoded_image)
        axes[i].axis("off")  # Hide axes
        if titles and i < len(titles):  # Set individual image titles if available
            axes[i].set_title(titles[i], fontsize=10)

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit main title
    plt.show()


def darken_fraction_of_image(
    image: np.ndarray, show_fraction: float, multiplier: float = 0.7
) -> np.ndarray:
    H, W, C = image.shape
    boundary_index = int(H * W * show_fraction)
    flat_image = image.reshape(-1, C)
    flat_image[boundary_index:] = (multiplier * flat_image[boundary_index:]).astype(
        image.dtype
    )
    return flat_image.reshape(H, W, C)


def plot_losses(
    train_losses: list[float] | list[np.ndarray],
    *,
    test_losses: list[float] | list[np.ndarray],
    save_dir: Path | None = None,
    show: bool = False,
    labels: str | list[str] = "Loss",
) -> None:
    assert type(train_losses[0]) is type(test_losses[0])
    if isinstance(labels, str):
        label_train: str | list[str] = f"Train {labels}"
        label_test: str | list[str] = f"Test {labels}"
    if isinstance(train_losses[0], np.ndarray) and isinstance(labels, list):
        assert train_losses[0].shape[0] == len(labels)  # type: ignore
        label_train = [f"Train {label}" for label in labels]
        label_test = [f"Test {label}" for label in labels]

    num_epochs = len(test_losses) - 1
    num_batches_total = len(train_losses)
    assert num_batches_total % num_epochs == 0

    x_train = np.linspace(0, num_epochs, len(train_losses))
    x_test = np.arange(num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x_train, train_losses, label=label_train)
    plt.plot(x_test, test_losses, label=label_test)

    plt.xlabel("Epochs")
    plt.ylabel("Loss (value at the end of the epoch)")
    plt.title("Training and Testing Losses")
    plt.legend()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "losses.png", bbox_inches="tight")
    if show:
        plt.show()
