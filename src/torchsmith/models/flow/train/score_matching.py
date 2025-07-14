import torch

from torchsmith.models.flow.paths.conditional_probability_path import (
    ConditionalProbabilityPath,
)
from torchsmith.models.flow.processes import SDE
from torchsmith.models.flow.train.base_trainer import FlowMatchingTrainer


class LearnedLangevinFlowSDE(SDE):
    """Langevin Flow SDE.

    .. math::
        dX_t = [ u_t^{\\theta}(x) + 0.5 * \sigma^2 s_t^{\\theta}(x) ] dt + \sigma dW_t
    """

    def __init__(
        self, flow_model: torch.nn.Module, *, score_model: torch.nn.Module, sigma: float
    ) -> None:
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.expand(x_t.shape[0], 1)  # Ensure t has the same batch size as x_t
        return self.flow_model(x_t, t) + 0.5 * self.sigma**2 * self.score_model(x_t, t)

    def diffusion_coefficient(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.randn_like(x_t)


class ConditionalScoreMatchingTrainer(FlowMatchingTrainer):
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

        s_theta = self.model(x, t)  # (num_samples, dim)
        s_ref = self.path.conditional_score(x, z, t)  # (num_samples, dim)

        return torch.nn.functional.mse_loss(s_theta, s_ref)
