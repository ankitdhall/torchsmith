from pathlib import Path
from typing import Callable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchsmith.tokenizers import TextTokenizer
from torchsmith.tokenizers.mnist_tokenizer import ColoredMNISTImageAndTextTokenizer
from torchsmith.tokenizers.mnist_tokenizer import VQVAEColoredMNISTImageTokenizer
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.scheduler import get_scheduler
from torchsmith.training.scheduler.base import BaseScheduler
from torchsmith.utils.dtypes import GenerateSamplesProtocol
from torchsmith.utils.pytorch import get_device

device = get_device()


def loop_AR(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    show_progress: bool = False,
) -> tuple[float, list[float]]:
    losses_per_batch = []
    loss_total = 0.0
    num_items = 0

    if optimizer is not None:
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Testing"

    dataloader = tqdm(dataloader, desc=desc) if show_progress else dataloader
    for X in dataloader:
        batch_size = X.shape[0]
        num_items += batch_size

        # Compute prediction and loss
        X = X.to(device)
        pred = model(X)
        loss = loss_fn(pred[:, :-1], X[:, 1:])

        losses_per_batch.append(loss.item())
        loss_total += loss.item() * batch_size

        if optimizer:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    loss_total = loss_total / num_items
    return loss_total, losses_per_batch


class TrainerAutoregression:
    def __init__(
        self,
        *,
        tokenizer: TextTokenizer
        | VQVAEColoredMNISTImageTokenizer
        | ColoredMNISTImageAndTextTokenizer,
        data_handler: DataHandler,
        train_config: TrainConfig,
        transformer,
        sequence_length: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        save_dir: Path | str,
        save_every_n_epochs: int,
        generate_samples_fn: GenerateSamplesProtocol,
        show_plots: bool = True,
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

        self.tokenizer = tokenizer
        self.data_handler = data_handler
        self.train_config = train_config
        self.seq_len = sequence_length
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.generate_samples_fn = generate_samples_fn
        self.show_plots = show_plots
        self.sample_every_n_epochs = sample_every_n_epochs
        self._epoch = 0

        # Prepare objects for training.
        self.transformer = transformer.to(device)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.train_config.optimizer_config.lr,
            weight_decay=self.train_config.optimizer_config.weight_decay,
        )
        if self.train_config.scheduler_config is not None:
            self.scheduler: BaseScheduler | None = get_scheduler(
                self.train_config.scheduler_config,
                optimizer=self.optimizer,
                epochs=self.train_config.num_epochs,
                num_batches_per_epoch=self.num_batches_per_epoch,
                dataset_len=self.train_dataset_len,
                batch_size=self.train_config.batch_size,
            )
        if self.show_plots and self.scheduler is not None:
            self.scheduler.visualize()

    def train(self) -> tuple:
        self.transformer.train()
        train_losses, test_losses = [], []
        train_dataloader = self.data_handler.get_dataloader("train")
        test_dataloader = self.data_handler.get_dataloader("test")
        print(
            f"Training starting from epoch: {self._epoch} to epoch: "
            f"{self.train_config.num_epochs}"
        )
        loss_total_test, _ = loop_AR(
            test_dataloader,
            self.transformer,
            self.loss_fn,
            optimizer=None,
            show_progress=False,
        )
        test_losses.append(loss_total_test)
        print(f"[At Epoch {self._epoch}] test: {loss_total_test: .4f}")
        for t in tqdm(range(self._epoch, self.train_config.num_epochs)):
            loss_total_train, losses_per_batch_train = loop_AR(
                train_dataloader,
                self.transformer,
                self.loss_fn,
                self.optimizer,
                scheduler=self.scheduler,
                show_progress=False,
            )
            train_losses.extend(losses_per_batch_train)

            loss_total_test, _ = loop_AR(
                test_dataloader,
                self.transformer,
                self.loss_fn,
                optimizer=None,
                show_progress=False,
            )
            test_losses.append(loss_total_test)

            print(
                f"[At Epoch {t + 1}] "
                f"train: {loss_total_train: .4f} "
                f"test: {loss_total_test: .4f}"
            )
            self._epoch += 1

            if self.save_every_n_epochs is not None and (
                (t + 1) % self.save_every_n_epochs == 0
            ):
                self.save_state(t + 1)

            if self.sample_every_n_epochs is not None and (
                (t + 1) % self.sample_every_n_epochs == 0 or t == 0
            ):
                samples = self.generate_samples_fn(
                    seq_len=self.seq_len,
                    tokenizer=self.tokenizer,
                    transformer=self.transformer,
                    decode=True,
                )

        samples = self.generate_samples_fn(
            seq_len=self.seq_len,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            decode=True,
        )
        print("Training complete!")

        return self.transformer, train_losses, test_losses, samples

    def save_state(self, epoch: int) -> None:
        dir_to_save = self.save_dir / f"epoch_{epoch}"
        dir_to_save.mkdir(parents=True)
        self.transformer.save_model(dir_to_save / "model.pth")
        data = {
            "epoch": self._epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        data.update(
            {"scheduler_state_dict": self.scheduler.state_dict()}
            if self.scheduler is not None
            else {}
        )
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
        self.transformer = self.transformer.load_model(dir_to_load / "model.pth")
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.train_config.optimizer_config.lr,
            weight_decay=self.train_config.optimizer_config.weight_decay,
        )
        if self.train_config.scheduler_config is not None:
            self.scheduler = get_scheduler(
                self.train_config.scheduler_config,
                optimizer=self.optimizer,
                epochs=self.train_config.num_epochs,
                num_batches_per_epoch=self.num_batches_per_epoch,
                train_dataset_len=self.train_dataset_len,
                batch_size=self.train_config.batch_size,
            )
        else:
            self.scheduler = None

        optimizer_state = torch.load(dir_to_load / "optimizer.pth", weights_only=False)
        self._epoch = optimizer_state["epoch"]
        self.optimizer.load_state_dict(optimizer_state["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(optimizer_state["scheduler_state_dict"])
        for param_group in self.optimizer.param_groups:
            print(f"Resuming with learning rate: {param_group['lr']}")
        print(f"Loaded trainer from {dir_to_load}")
