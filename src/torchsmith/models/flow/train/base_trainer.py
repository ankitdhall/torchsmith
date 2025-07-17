from abc import ABC
from abc import abstractmethod

import torch
from tqdm import tqdm

from torchsmith.utils.pytorch import get_device

device = get_device()


class FlowMatchingTrainer(ABC):
    def __init__(
        self, model: torch.nn.Module, num_epochs: int, batch_size: int, lr: float = 1e-3
    ) -> None:
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @abstractmethod
    def get_train_loss(self) -> torch.Tensor:
        raise NotImplementedError()

    def train(self) -> tuple[torch.nn.Module, list[float]]:
        self.model.to(device)
        self.model.train()

        train_losses = [self.get_train_loss().item()]
        pbar = tqdm(range(self.num_epochs))
        for epoch_index in pbar:
            self.optimizer.zero_grad()
            loss = self.get_train_loss()
            train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f"Epoch {epoch_index}, loss: {loss.item()}")

        return self.model, train_losses
