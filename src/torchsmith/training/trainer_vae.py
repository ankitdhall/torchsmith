import numpy as np
import torch
from torch.optim import Optimizer
from tqdm import tqdm

from torchsmith.training.trainer_vae_conv import loop_VAE
from torchsmith.utils.pytorch import get_device

device = get_device()


# TODO: remove and replace by VAETrainer
def train_model_vae(
    *,
    num_epochs: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    test_dataloader: torch.utils.data.DataLoader,
    train_dataloader: torch.utils.data.DataLoader,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train_losses, test_losses = [], []
    loss_total_test, _ = loop_VAE(
        test_dataloader,
        model=model,
        optimizer=None,
        show_progress=False,
    )
    test_losses.append(loss_total_test)
    print(f"[At Epoch 0] test: {np.round(loss_total_test, 4)}")
    for t in tqdm(range(num_epochs)):
        loss_total_train, losses_per_batch_train = loop_VAE(
            train_dataloader,
            model=model,
            optimizer=optimizer,
            show_progress=False,
        )
        train_losses.extend(losses_per_batch_train)

        loss_total_test, _ = loop_VAE(
            test_dataloader,
            model=model,
            optimizer=None,
            show_progress=False,
        )
        test_losses.append(loss_total_test)

        print(
            f"[At Epoch {t + 1}] "
            f"train: {np.round(loss_total_train, 4)} "
            f"test: {np.round(loss_total_test, 4)}"
        )
    return test_losses, train_losses
