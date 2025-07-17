from matplotlib import pyplot as plt

from torchsmith.models.flow.data import Gaussian
from torchsmith.models.flow.data.exotic import CheckerboardSampleable
from torchsmith.models.flow.paths.linear_conditional_probability_path import (
    LinearConditionalProbabilityPath,
)
from torchsmith.models.flow.solvers import EulerSolver
from torchsmith.models.flow.train.flow_matching import ConditionalFlowMatchingTrainer
from torchsmith.models.flow.train.flow_matching import LearnedVectorFieldODE
from torchsmith.models.flow.train.model import MLPForMatching
from torchsmith.models.flow.visualize import visualize_conditional_probability_path
from torchsmith.models.flow.visualize import visualize_field_across_time_and_space
from torchsmith.models.flow.visualize import visualize_marginal_probability_path
from torchsmith.models.flow.visualize import visualize_samples_from_learned_marginal
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device

device = get_device()
PLOT_LIMITS = 10.0

##########################################
# Construct conditional probability path #
##########################################

p_source = Gaussian.isotropic(dim=2, std=1.0)
p_target = CheckerboardSampleable(grid_size=4)
path = LinearConditionalProbabilityPath(
    p_source=p_source,
    p_target=p_target,
).to(device)


z = path.sample_conditioning_variable(1)
visualize_conditional_probability_path(
    path=path, z=z, plot_limits=PLOT_LIMITS, num_samples=2000
)
plt.show()

visualize_marginal_probability_path(
    path=path, plot_limits=PLOT_LIMITS, num_samples=2000
)
plt.show()

############
# Training #
############


NUM_EPOCHS = 20000
flow_model = MLPForMatching(num_dims=2, hidden_dims=[100] * 4).to(device)

trainer = ConditionalFlowMatchingTrainer(
    path, flow_model, num_epochs=NUM_EPOCHS, lr=1e-3, batch_size=1000
)
flow_model, train_losses = trainer.train()
plot_losses(train_losses=train_losses[1:], test_losses=train_losses, show=True)

learned_ode = LearnedVectorFieldODE(flow_model.eval())
solver = EulerSolver(learned_ode)


##################
# Visualizations #
##################


visualize_samples_from_learned_marginal(
    path=path, plot_limits=PLOT_LIMITS, solver=solver
)
plt.show()

##################################
# Visualize Learned Vector Field #
##################################


ax = visualize_field_across_time_and_space(
    flow_model,
    path=path,
    num_marginals=5,
    plot_limits=PLOT_LIMITS,
    num_bins=60,
    title="Learned Vector Field",
)
plt.show()
