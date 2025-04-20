from typing import Literal

import numpy as np
import torchvision
from torchvision import transforms

from torchsmith.utils.constants import DATA_DIR


def preprocess_data(x: np.ndarray) -> np.ndarray:
    # Assume x in [0, 1]
    x = x - 0.5  # [0, 1] -> [-0.5, 0.5]
    x = x * 2  # [-0.5, 0.5] -> [-1, 1]
    return x  # in [-1, 1]


def postprocess_data(x: np.ndarray) -> np.ndarray:
    # Assume x in [-1, 1]
    x = np.clip(x, a_min=-1, a_max=1)
    x = (x / 2) + 0.5  # -> [0, 1]
    x = (x * 255).astype(int)
    # x = np.transpose(x, (0, 2, 3, 1))
    return x  # in [0, 255]


def get_svhn(split: Literal["train", "test"]) -> np.ndarray:
    dataset = torchvision.datasets.SVHN(
        root=DATA_DIR / "svhn",
        split=split,
        download=True,
        transform=transforms.ToTensor(),
    )
    data = dataset.data.transpose((0, 2, 3, 1))

    print("Before pre-processing ...")
    print(
        f"'{split}' dataset: \nshape: {data.shape}, "
        f"min: {np.min(data)}, max: {np.max(data)}"
    )

    data = (np.transpose(data, (0, 3, 1, 2)) / 255.0).astype("float32")
    data = preprocess_data(data)

    print("After pre-processing ...")
    print(
        f"'{split}' dataset: \nshape: {data.shape}, "
        f"min: {np.min(data)}, max: {np.max(data)}"
    )
    return data
