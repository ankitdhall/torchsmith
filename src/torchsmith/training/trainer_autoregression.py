import json
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from torchsmith.tokenizers import TextTokenizer
from torchsmith.tokenizers.mnist_tokenizer import ColoredMNISTImageAndTextTokenizer
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.training.config import TrainConfig
from torchsmith.training.config import WandbConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.scheduler import get_scheduler
from torchsmith.training.scheduler.base import BaseScheduler
from torchsmith.utils.dtypes import GenerateSamplesProtocol


def loop_AR(
    *,
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable,
    accelerator: Accelerator,
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
        pred = model(X)
        with accelerator.autocast():
            loss = loss_fn(pred[:, :-1], X[:, 1:])

        losses_per_batch.append(loss.item())
        loss_total += loss.item() * batch_size

        if optimizer:
            accelerator.backward(loss)
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
        | VQVAEImageTokenizer
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
        wandb_config: WandbConfig | None = None,
        train_dataset_len: int | None = None,
        sample_every_n_epochs: int | None = None,
    ) -> None:
        self.wandb_config = wandb_config
        self.accelerator = (
            Accelerator(log_with="wandb") if self.wandb_config else Accelerator()
        )

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
        self.transformer = transformer
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.train_config.optimizer_config.lr,
            weight_decay=self.train_config.optimizer_config.weight_decay,
        )
        (
            self.transformer,
            self.optimizer,
        ) = self.accelerator.prepare(self.transformer, self.optimizer)

        if self.train_config.scheduler_config is not None:
            self.scheduler: BaseScheduler | None = get_scheduler(
                self.train_config.scheduler_config,
                optimizer=self.optimizer,
                epochs=self.train_config.num_epochs,
                num_batches_per_epoch=self.num_batches_per_epoch,
                dataset_len=self.train_dataset_len,
                batch_size=self.train_config.batch_size,
            )
            if self.show_plots:
                self.scheduler.visualize()

            self.scheduler = self.accelerator.prepare(self.scheduler)

        if self.wandb_config:
            self.accelerator.init_trackers(
                self.wandb_config.project_name, config=self.wandb_config.config
            )

    def log_samples(self, samples: list[Any], epoch: int) -> None:
        sample_table = wandb.Table(columns=["sample", "sample_id", "epoch"])  # type: ignore
        for sample_id, sample in enumerate(samples):
            if isinstance(sample, str):
                sample_table.add_data(sample, sample_id, epoch)
            elif isinstance(sample, np.ndarray):
                sample_table.add_data(wandb.Image(sample), sample_id, epoch)  # type: ignore
            else:
                raise TypeError(
                    f"Unsupported sample type: '{type(sample)}'. "
                    "Expected str or np.ndarray."
                )
        wandb.log({f"samples_{epoch}": sample_table}, step=epoch)  # type: ignore

    def train(self) -> tuple:
        self.transformer.train()
        train_losses, test_losses = [], []
        train_dataloader = self.accelerator.prepare(
            self.data_handler.get_dataloader("train")
        )
        test_dataloader = self.accelerator.prepare(
            self.data_handler.get_dataloader("test")
        )
        if self.wandb_config and self.wandb_config.watch_config is not None:
            wandb.watch(  # type: ignore
                self.transformer,
                log=self.wandb_config.watch_config.log,
                log_graph=self.wandb_config.watch_config.log_graph,
                log_freq=self.wandb_config.watch_config.log_freq,
            )
        print(
            f"Training starting from epoch: {self._epoch} to epoch: "
            f"{self.train_config.num_epochs}"
        )
        loss_total_test, _ = loop_AR(
            dataloader=test_dataloader,
            model=self.transformer,
            loss_fn=self.loss_fn,
            accelerator=self.accelerator,
            optimizer=None,
            show_progress=False,
        )
        test_losses.append(loss_total_test)
        print(f"[At Epoch {self._epoch}] test: {loss_total_test: .4f}")
        self.accelerator.log({"test_loss": loss_total_test}, step=self._epoch)
        for t in tqdm(range(self._epoch, self.train_config.num_epochs)):
            loss_total_train, losses_per_batch_train = loop_AR(
                dataloader=train_dataloader,
                model=self.transformer,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                show_progress=False,
            )
            train_losses.extend(losses_per_batch_train)
            self.accelerator.log({"train_loss": loss_total_train}, step=self._epoch + 1)

            # Assumes, logs only the first group.
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.accelerator.log({"learning_rate": current_lr}, step=self._epoch + 1)

            loss_total_test, _ = loop_AR(
                dataloader=test_dataloader,
                model=self.transformer,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                optimizer=None,
                show_progress=False,
            )
            self.accelerator.log({"test_loss": loss_total_test}, step=self._epoch + 1)
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
                retval = self.generate_samples_fn(
                    seq_len=self.seq_len,
                    tokenizer=self.tokenizer,
                    transformer=self.transformer,
                    decode=True,
                )
                samples = retval[1] if isinstance(retval, tuple) else retval
                if self.wandb_config:
                    self.log_samples(samples, self._epoch + 1)

        retval = self.generate_samples_fn(
            seq_len=self.seq_len,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            decode=True,
        )

        samples = retval[1] if isinstance(retval, tuple) else retval
        if self.wandb_config:
            self.log_samples(samples, self._epoch + 1)
        print("Training complete!")

        return self.transformer, train_losses, test_losses, samples

    def save_state(self, epoch: int) -> None:
        dir_to_save = self.save_dir / f"epoch_{epoch}"
        dir_to_save.mkdir(parents=True)

        # Save model.
        self.accelerator.wait_for_everyone()

        # Save state.
        self.accelerator.save_state(str(dir_to_save / "state"))

        with open(dir_to_save / "info.json", "w") as f:
            json.dump({"epoch": self._epoch}, f, indent=4)

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

        # Load state.
        self.accelerator.load_state(str(dir_to_load / "state"))
        with open(dir_to_load / "info.json") as f:
            info = json.load(f)
            self._epoch = info.get("epoch", 0)

        print(f"Loaded trainer from {dir_to_load}")
