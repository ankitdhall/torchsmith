from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchsmith.models.vae.base import BaseVAE
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.utils.pytorch import get_device

device = get_device()


def loop_VAE(
    dataloader: DataLoader,
    *,
    model: BaseVAE,
    optimizer: Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    history: defaultdict[str, list] = defaultdict(list)

    if optimizer is not None:
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Testing"

    dataloader = tqdm(dataloader, desc=desc) if show_progress else dataloader
    for X in dataloader:
        batch_size = X.shape[0]

        # Compute prediction and loss
        if not optimizer:
            with torch.no_grad():
                X = X.to(device)
                loss = model.loss(X)
        else:
            X = X.to(device)
            loss = model.loss(X)

        for key in loss.KEYS:
            history[key].append(getattr(loss, key).item())
        history["num_items"].append(batch_size)

        if optimizer:
            loss.get_total_loss().backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    num_items_total = sum(history["num_items"])
    loss_total = np.array([sum(history[key]) / num_items_total for key in loss.KEYS])
    losses_per_batch = np.stack(
        [np.array(history[key]) / np.array(history["num_items"]) for key in loss.KEYS],
        axis=-1,
    )
    loss_description = loss.DESCRIPTION
    return loss_total, losses_per_batch, loss_description


class VAETrainer:
    def __init__(
        self,
        *,
        model: BaseVAE,
        data_handler: DataHandler,
        train_config: TrainConfig,
        generate_samples_fn: Callable,
        save_dir: Path | str,
        save_every_n_epochs: int,
        train_dataset_len: int | None = None,
        sample_every_n_epochs: int | None = None,
    ) -> None:
        self.train_dataset_len = train_dataset_len
        self.num_batches_per_epoch = data_handler.get_length("train")
        if self.train_dataset_len is None and self.num_batches_per_epoch is None:
            raise ValueError(
                "Cannot determine train dataset length. "
                "Please provide it explicitly by passing `train_dataset_len`."
            )

        self.data_handler = data_handler
        self.generate_samples_fn = generate_samples_fn
        self.train_config = train_config
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.sample_every_n_epochs = sample_every_n_epochs
        self._epoch = 0

        # Prepare objects for training.
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.optimizer_config.lr,
            weight_decay=self.train_config.optimizer_config.weight_decay,
        )

    def train(self) -> tuple:
        self.model.train()
        train_losses, test_losses = [], []
        train_dataloader = self.data_handler.get_dataloader("train")
        test_dataloader = self.data_handler.get_dataloader("test")
        print(
            f"Training starting from epoch: {self._epoch} to epoch: "
            f"{self.train_config.num_epochs}"
        )
        loss_total_test, _, LOSS_DESCRIPTION = loop_VAE(
            test_dataloader,
            model=self.model,
            optimizer=None,
            show_progress=False,
        )
        test_losses.append(loss_total_test)
        print(
            f"[At Epoch {self._epoch}] "
            f"test: {LOSS_DESCRIPTION}: {np.round(loss_total_test, 4)}"
        )
        for t in tqdm(range(self._epoch, self.train_config.num_epochs)):
            loss_total_train, losses_per_batch_train, LOSS_DESCRIPTION = loop_VAE(
                train_dataloader,
                model=self.model,
                optimizer=self.optimizer,
                show_progress=False,
            )
            train_losses.extend(losses_per_batch_train)

            loss_total_test, _, LOSS_DESCRIPTION = loop_VAE(
                test_dataloader,
                model=self.model,
                optimizer=None,
                show_progress=False,
            )
            test_losses.append(loss_total_test)

            print(
                f"[At Epoch {t + 1}] "
                f"train: {LOSS_DESCRIPTION}: {np.round(loss_total_train, 4)} "
                f"test: {LOSS_DESCRIPTION}: {np.round(loss_total_test, 4)}"
            )
            self._epoch += 1

            if self.save_every_n_epochs is not None and (
                (t + 1) % self.save_every_n_epochs == 0
            ):
                self.save_state(t + 1)

            if self.sample_every_n_epochs is not None and (
                (t + 1) % self.sample_every_n_epochs == 0 or t == 0
            ):
                samples = self.generate_samples_fn(self.model)

        samples = self.generate_samples_fn(self.model)
        print("Training complete!")

        return self.model, train_losses, test_losses, samples

    def save_state(self, epoch: int) -> None:
        dir_to_save = self.save_dir / f"epoch_{epoch}"
        dir_to_save.mkdir(parents=True)
        self.model.save_model(dir_to_save / "model.pth")
        data = {
            "epoch": self._epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(
            data,
            dir_to_save / "optimizer.pth",
        )

    def load_state(self) -> None:
        assert self.save_dir.exists()
        latest_epoch = sorted(
            [
                int(p.name.split("epoch_")[-1])
                for p in self.save_dir.iterdir()
                if p.name.startswith("epoch_")
            ]
        )[-1]
        dir_to_load = self.save_dir / f"epoch_{latest_epoch}"
        self.model = self.model.load_model(dir_to_load / "model.pth").to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.optimizer_config.lr,
            weight_decay=self.train_config.optimizer_config.weight_decay,
        )
        optimizer_state = torch.load(dir_to_load / "optimizer.pth", weights_only=False)
        self._epoch = optimizer_state["epoch"]
        self.optimizer.load_state_dict(optimizer_state["optimizer_state_dict"])
        for param_group in self.optimizer.param_groups:
            print(f"Resuming with learning rate: {param_group['lr']}")
        print(f"Loaded trainer from {dir_to_load}")
