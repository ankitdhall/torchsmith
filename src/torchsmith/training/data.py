import dataclasses
from typing import Literal
from typing import cast

import torch
from torch.utils.data import DataLoader

from torchsmith.training.config import TrainConfig


@dataclasses.dataclass
class DataHandler:
    train_config: TrainConfig
    train_dataset: torch.utils.data.Dataset | None = None
    test_dataset: torch.utils.data.Dataset | None = None
    train_dataloader: torch.utils.data.DataLoader | None = None
    test_dataloader: torch.utils.data.DataLoader | None = None

    def __post_init__(self) -> None:
        self.dataset_available = False
        if self.train_dataset is not None and self.test_dataset is not None:
            self.dataset_available = True
        self.dataloader_available = False
        if self.train_dataloader is not None and self.test_dataloader is not None:
            self.dataloader_available = True

        if not (self.dataloader_available ^ self.dataset_available):
            raise ValueError(
                "Either pass dataset or dataloader but not both and not neither."
            )

        if not self.dataloader_available:
            train_dataset = cast(torch.utils.data.dataset, self.train_dataset)
            test_dataset = cast(torch.utils.data.dataset, self.test_dataset)
            print(f"Train dataset {len(train_dataset)}")
            print(f"Test dataset {len(test_dataset)}")

            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                num_workers=self.train_config.num_workers,
                pin_memory=True,
            )
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                num_workers=self.train_config.num_workers,
                pin_memory=True,
            )

    def get_dataloader(
        self, split: Literal["train", "test"]
    ) -> torch.utils.data.DataLoader:
        valid_splits = {"train", "test"}
        if split not in valid_splits:
            raise ValueError(
                f"'{split}' is not a valid split. Valid splits are: {valid_splits}."
            )
        dataloader = self.train_dataloader if split == "train" else self.test_dataloader
        return dataloader

    def get_length(self, split: Literal["train", "test"]) -> int | None:
        try:
            dataloader = self.get_dataloader(split=split)
            length = len(dataloader)
            print(f"Length of '{split}' dataloader: {length}")  # type: ignore
            return length
        except Exception as e:
            print(f"Cannot print dataloader length for '{split}': ", str(e))
            return None
