import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from torchsmith.models.diffusion.diffusion import DiffusionModel
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.scheduler import get_scheduler
from torchsmith.training.scheduler.base import BaseScheduler


def loop_diffusion(
    dataloader: DataLoader,
    *,
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
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            X, *rest = batch
            kwargs = rest[0] if rest else {}  # Get kwargs if available, else empty dict
        else:
            X, kwargs = batch, {}

        batch_size = X.shape[0]
        num_items += batch_size

        # Compute prediction and loss
        X = X
        kwargs = {
            k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        y_hat, y = model(X, **kwargs)
        with accelerator.autocast():
            loss = loss_fn(y_hat, y)

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


class DiffusionTrainer:
    def __init__(
        self,
        *,
        model: DiffusionModel,
        data_handler: DataHandler,
        train_config: TrainConfig,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        generate_samples_fn: Callable,
        save_dir: Path | str,
        save_every_n_epochs: int,
        show_plots: bool = True,
        train_dataset_len: int | None = None,
        sample_every_n_epochs: int | None = None,
        wandb_project_name: str | None = None,
    ) -> None:
        self.use_wandb = wandb_project_name is not None
        self.accelerator = (
            Accelerator(log_with="wandb") if self.use_wandb else Accelerator()
        )

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
        self.show_plots = show_plots
        self.sample_every_n_epochs = sample_every_n_epochs
        self._epoch = 0
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.optimizer_config.lr,
            weight_decay=self.train_config.optimizer_config.weight_decay,
        )
        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(self.model, self.optimizer)
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
        else:
            self.scheduler = None
        if self.use_wandb:
            hps = {}
            for config in [self.train_config]:
                hps[config.__class__.__name__] = asdict(config)
            self.accelerator.init_trackers(wandb_project_name, config=hps)

    def log_samples(self, samples: list, epoch: int) -> None:
        sample_table = wandb.Table(columns=["sample", "sample_id", "epoch"])  # type: ignore
        for sample_id, sample in enumerate(samples):
            if isinstance(sample, str):
                sample_table.add_data(sample, sample_id, epoch)
            elif isinstance(sample, np.ndarray):
                if sample.ndim == 3 or sample.ndim == 2:
                    sample_table.add_data(wandb.Image(sample), sample_id, epoch)  # type: ignore
                elif sample.ndim == 4:
                    b, c, h, w = sample.shape
                    grid_img = make_grid(torch.tensor(sample), nrow=b // 10)
                    sample_table.add_data(wandb.Image(grid_img), sample_id, epoch)  # type: ignore
                else:
                    raise NotImplementedError(
                        f"Unsupported sample shape: {sample.shape}"
                    )
            else:
                raise TypeError(
                    f"Unsupported sample type: '{type(sample)}'. "
                    "Expected str or np.ndarray."
                )
        wandb.log({f"samples_{epoch}": sample_table}, step=epoch)  # type: ignore

    def train(self) -> tuple:
        self.model.train()
        train_losses, test_losses = [], []
        train_dataloader = self.accelerator.prepare(
            self.data_handler.get_dataloader("train")
        )
        test_dataloader = self.accelerator.prepare(
            self.data_handler.get_dataloader("test")
        )

        if self.use_wandb:
            wandb.watch(self.model, log="all", log_graph=True, log_freq=10)  # type: ignore
        print(
            f"Training starting from epoch: {self._epoch} to epoch: "
            f"{self.train_config.num_epochs}"
        )
        loss_total_test, _ = loop_diffusion(
            test_dataloader,
            model=self.model,
            loss_fn=self.loss_fn,
            accelerator=self.accelerator,
            optimizer=None,
            show_progress=False,
        )
        test_losses.append(loss_total_test)
        print(f"[At Epoch {self._epoch}] test: {loss_total_test: .4f}")
        if self.use_wandb:
            self.accelerator.log({"test_loss": loss_total_test}, step=self._epoch)

        for t in tqdm(range(self._epoch, self.train_config.num_epochs)):
            loss_total_train, losses_per_batch_train = loop_diffusion(
                train_dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                show_progress=False,
            )

            train_losses.extend(losses_per_batch_train)
            if self.use_wandb:
                self.accelerator.log(
                    {"train_loss": loss_total_train}, step=self._epoch + 1
                )

            if self.use_wandb:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.accelerator.log(
                    {"learning_rate": current_lr}, step=self._epoch + 1
                )

            loss_total_test, _ = loop_diffusion(
                test_dataloader,
                model=self.model,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                optimizer=None,
                show_progress=False,
            )

            if self.use_wandb:
                self.accelerator.log(
                    {"test_loss": loss_total_test}, step=self._epoch + 1
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
                samples = self.generate_samples_fn(self.model)
                if self.use_wandb:
                    self.log_samples([samples], self._epoch + 1)

        samples = self.generate_samples_fn(self.model)
        if self.use_wandb:
            self.log_samples([samples], self._epoch + 1)

        print("Training complete!")
        return self.model, train_losses, test_losses, samples

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
        self.accelerator.load_state(str(dir_to_load / "state"), {"weights_only": False})
        with open(dir_to_load / "info.json") as f:
            info = json.load(f)
            self._epoch = info.get("epoch", 0)

        print(f"Loaded trainer from {dir_to_load}")
