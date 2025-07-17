import torch

from torchsmith.models.flow.paths.conditional_probability_path import (
    ConditionalProbabilityPath,
)
from torchsmith.models.flow.processes import ODE
from torchsmith.models.flow.train.base_trainer import FlowMatchingTrainer
from torchsmith.utils.pytorch import get_device

device = get_device()


class LearnedVectorFieldODE(ODE):
    def __init__(self, vector_field: torch.nn.Module) -> None:
        super().__init__()
        self.vector_field = vector_field

    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.expand(x_t.shape[0], 1)  # Ensure t has the same batch size as x_t
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        assert x_t.dim() == 2 and t.dim() == 2, (
            f"x_t and t must be 2D tensors but were x_t: {x_t.shape} and t: {t.shape}."
        )
        return self.vector_field(x_t, t)


class ConditionalFlowMatchingTrainer(FlowMatchingTrainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: torch.nn.Module,
        num_epochs: int,
        batch_size: int,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(model, num_epochs=num_epochs, batch_size=batch_size, lr=lr)
        self.path = path

    def get_train_loss(self) -> torch.Tensor:
        z = self.path.p_target.sample(self.batch_size)  # (num_samples, dim)
        t = torch.rand(self.batch_size, 1)  # (num_samples, 1)
        x = self.path.sample_conditional_path(z, t)  # (num_samples, dim)

        u_theta = self.model(x, t)  # (num_samples, dim)
        u_ref = self.path.conditional_vector_field(x, z, t)  # (num_samples, dim)

        return torch.nn.functional.mse_loss(u_theta, u_ref)
