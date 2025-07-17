from matplotlib import pyplot as plt

from torchsmith.models.flow.data import Gaussian
from torchsmith.models.flow.data import GaussianMixture
from torchsmith.models.flow.paths.gaussian_conditional_probability_path import (
    GaussianConditionalProbabilityPath,
)
from torchsmith.models.flow.paths.schedulers import LinearAlpha
from torchsmith.models.flow.paths.schedulers import SquareRootBeta
from torchsmith.models.flow.solvers import EulerMaruyamaSolver
from torchsmith.models.flow.train.flow_matching import ConditionalFlowMatchingTrainer
from torchsmith.models.flow.train.model import MLPForMatching
from torchsmith.models.flow.train.score_matching import ConditionalScoreMatchingTrainer
from torchsmith.models.flow.train.score_matching import LearnedLangevinFlowSDE
from torchsmith.models.flow.train.score_matching import ScoreFromVectorField
from torchsmith.models.flow.visualize import visualize_density
from torchsmith.models.flow.visualize import visualize_field_across_time_and_space
from torchsmith.models.flow.visualize import visualize_generated_trajectories
from torchsmith.models.flow.visualize import (
    visualize_marginal_probability_path_overlaid,
)
from torchsmith.models.flow.visualize import (
    visualize_samples_from_learned_marginal_overlaid,
)
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
    p_source=p_source, p_target=p_data, alpha=LinearAlpha(), beta=SquareRootBeta()
).to(device)

NUM_EPOCHS = 1000
##############
# Flow Model #
##############

# Construct learnable vector field
flow_model = MLPForMatching(num_dims=2, hidden_dims=[64, 64, 64, 64]).to(device)

# Construct trainer
flow_matching_trainer = ConditionalFlowMatchingTrainer(
    path, flow_model, num_epochs=NUM_EPOCHS, lr=1e-3, batch_size=1000
)
flow_model, train_losses = flow_matching_trainer.train()
plot_losses(train_losses=train_losses[1:], test_losses=train_losses, show=True)

###############
# Score Model #
###############

# Construct learnable score
score_model = MLPForMatching(num_dims=2, hidden_dims=[64, 64, 64, 64]).to(device)

# Construct trainer
score_matching_trainer = ConditionalScoreMatchingTrainer(
    path, score_model, num_epochs=NUM_EPOCHS, lr=1e-3, batch_size=1000
)
score_model, train_losses = score_matching_trainer.train()
plot_losses(train_losses=train_losses[1:], test_losses=train_losses, show=True)

################
# Langevin SDE #
################
SIGMA = 2.0
sde = LearnedLangevinFlowSDE(
    flow_model.eval(), score_model=score_model.eval(), sigma=SIGMA
)
solver = EulerMaruyamaSolver(sde)

fig, axes = plt.subplots(1, 3, figsize=(36, 12))
visualize_density(p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=axes[0])
visualize_marginal_probability_path_overlaid(
    path=path, plot_limits=plot_limits, num_samples=1000, ax=axes[0]
)
visualize_density(p_source=p_source, p_data=p_data, plot_limits=plot_limits, ax=axes[1])

visualize_samples_from_learned_marginal_overlaid(
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

#####################################################################
# Compare Learned Score and Score Derived From Learned Vector Field #
#####################################################################

score_model_from_vector_field = ScoreFromVectorField(
    vector_field=flow_model.eval(), alpha=path.alpha, beta=path.beta
).to(device)

ax = visualize_field_across_time_and_space(
    score_model,
    path=path,
    num_marginals=5,
    plot_limits=plot_limits,
    num_bins=40,
    title="Learned Score Field",
)
plt.show()

ax = visualize_field_across_time_and_space(
    score_model_from_vector_field,
    path=path,
    num_marginals=5,
    plot_limits=plot_limits,
    num_bins=40,
    title="Score Field Derived From Learned Vector Field",
)
plt.show()
