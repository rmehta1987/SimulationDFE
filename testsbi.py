from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import torch

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis

# 2 rounds: first round simulates from the prior, second round simulates parameter set
# that were sampled from the obtained posterior.
num_rounds = 2
# The specific observation we want to focus the inference on.
x_o = torch.zeros(
    3,
)

num_dim = 3
prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

posteriors = []
proposal = prior


def linear_gaussian(theta):
    return theta + 1.0 + torch.randn_like(theta) * 0.1

simulator, prior = prepare_for_sbi(linear_gaussian, prior)
inference = SNPE(prior=prior)


for _ in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)

    # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal
    ).train()
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)