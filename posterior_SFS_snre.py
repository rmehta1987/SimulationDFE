import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, SNLE, MNLE, SNRE, SNRE_A
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
import numpy as np
import moments
from matplotlib import pyplot as plt
from sbi.utils import BoxUniform
from sbi.utils import MultipleIndependent
from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding, FCEmbedding
from sbi.utils.user_input_checks import process_prior, process_simulator

import seaborn as sns

# Setup devices and constants
the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_size = 100 # Sample size of population, n
iid = 50  # number of independent samples, but with the same selection coefficeint (i.e. [P(X_1 | gamma_1), P(X_2 | gamma_1), ... , P(X_iid | gamma_1)])
prior_returns_numpy = True # Prior needs to be in numpy format for simulator
default_network_type = "maf" # TODO change default network to integer discrete flows
num_sim = 1000 # Number of simulations, number of simulations should increase when number of parameters in the simulation increase
rounds = 10 # Number of rounds to train/simulate

print("Using device: {}".format(the_device))

def moment_sim(prior: float) -> torch.float32:
    """Uses Moments (cite) to create simulated Site Frequency Spectrums of a demographic history

    Args:
        prior (float): a sampled selection distribution from a prior distribution

    Returns:
        x: The sampled site-frequency spectrum based on moments via Poisson(E[X | gamma]) where gamma = prior
        Output of the simulator needs to be float32 based on SBI format 
    """    
    
    moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(10, gamma=prior, theta=100.0))  # returns a masked array
    
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    actual_fs = moment_data.compressed()  
    x = torch.poisson(torch.tensor(actual_fs, device=the_device)).type(torch.float32)
        
    return x

# Create Prior Distribution(s)
# multi_prior = MultipleIndependent(
#     [
#         torch.distributions.Gamma(torch.tensor([100.0],device=the_device), torch.tensor([10.0],device=the_device)),
#         torch.distributions.Categorical(torch.tensor([1.0/sample_size],device=the_device).repeat(sample_size,0)),
#     ],
    
#     validate_args=False)
    
gamma_prior = torch.distributions.Gamma(torch.tensor([100.0],device=the_device), torch.tensor([10.0],device=the_device))
# Need to create independent distribution so event_shape is 1, see https://bochang.me/blog/posts/pytorch-distributions/ 
# and https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
ind_gamma_prior = torch.distributions.independent.Independent(gamma_prior, 1, validate_args=None)  

# Set up prior and simulator for SBI
prior, num_parameters, prior_returns_numpy = process_prior(ind_gamma_prior)
simulator = process_simulator(moment_sim, prior, prior_returns_numpy)

# Create inference procedure
inferer = SNRE_A(prior, show_progress_bars=True, device='cuda',)

#posterior parameters
vi_parameters = dict(q="maf")

proposal = prior
posteriors = []

true_x = moment_sim(prior.sample((1,)).cpu().numpy())

# Train posterior
for i in range(0,rounds):

    theta, x = simulate_for_sbi(simulator, proposal, num_sim)
    liklihood_estimator = inferer.append_simulations(theta, x).train(training_batch_size=50)
    posterior = inferer.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(true_x).train(max_num_iters=40, quality_control=False )

print("Finished Training posterior and prior")

posterior_samples = posterior.sample((10000,), x=true_x)
prior_samples = prior.sample((10000,))
sns.kdeplot(posterior_samples.detach().cpu().numpy())
plt.savefig('testpost.png')
plt.close()
sns.kdeplot(prior_samples.detach().cpu().numpy())
plt.savefig('testprior.png')
plt.close()

