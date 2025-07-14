from matplotlib import pyplot as plt

from torchsmith.models.flow.data import Gaussian
from torchsmith.models.flow.data import GaussianMixture
from torchsmith.models.flow.paths.gaussian_conditional_probability_path import (
    GaussianConditionalProbabilityPath,
)
from torchsmith.models.flow.paths.schedulers import LinearAlpha
from torchsmith.models.flow.paths.schedulers import SquareRootBeta
from torchsmith.models.flow.solvers import EulerSolver
from torchsmith.models.flow.train.flow_matching import ConditionalFlowMatchingTrainer
from torchsmith.models.flow.train.flow_matching import LearnedVectorFieldODE
from torchsmith.models.flow.train.model import MLPForMatching
from torchsmith.models.flow.visualize import visualize_density
from torchsmith.models.flow.visualize import visualize_generated_trajectories
from torchsmith.models.flow.visualize import visualize_marginal_probability_path
from torchsmith.models.flow.visualize import visualize_samples_from_learned_marginal
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device

device = get_device()

# Construct conditional probability path
target_scale = 10.0
target_std = 1.0
plot_limits = target_scale * 1.5
p_source = Gaussian.isotropic(dim=2, std=1.0).to(device)
p_data = GaussianMixture.symmetric_2d(
    num_modes=5,
    std=target_std,
    scale=target_scale,
).to(device)
path = GaussianConditionalProbabilityPath(
    p_source=p_source, p_data=p_data, alpha=LinearAlpha(), beta=SquareRootBeta()
).to(device)

NUM_EPOCHS = 1000
# Construct learnable vector field
flow_model = MLPForMatching(num_dims=2, hidden_dims=[64, 64, 64, 64]).to(device)

# Construct trainer
trainer = ConditionalFlowMatchingTrainer(
    path, flow_model, num_epochs=NUM_EPOCHS, lr=1e-3, batch_size=1000
)
flow_model, train_losses = trainer.train()
plot_losses(train_losses=train_losses[1:], test_losses=train_losses, show=True)

learned_ode = LearnedVectorFieldODE(flow_model.eval())
solver = EulerSolver(learned_ode)

fig, axes = plt.subplots(1, 3, figsize=(36, 12))
visualize_density(p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=axes[0])
visualize_marginal_probability_path(
    path=path, plot_limits=plot_limits, num_samples=1000, ax=axes[0]
)
visualize_density(p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=axes[1])

visualize_samples_from_learned_marginal(
    path=path, plot_limits=plot_limits, solver=solver, ax=axes[1]
)

visualize_density(p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=axes[2])
visualize_generated_trajectories(
    solver=solver,
    p_source=p_source,
    plot_limits=plot_limits,
    num_trajectories=200,
    ax=axes[2],
)
plt.show()
