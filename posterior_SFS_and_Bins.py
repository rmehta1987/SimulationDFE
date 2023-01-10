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
from sbi.utils import get_density_thresholder, RestrictedPrior
import pickle

import seaborn as sns

# Setup devices and constants
the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_size = 10 # Sample size of population, n
iid = 50  # number of independent samples, but with the same selection coefficeint (i.e. [P(X_1 | gamma_1), P(X_2 | gamma_1), ... , P(X_iid | gamma_1)])
prior_returns_numpy = True # Prior needs to be in numpy format for simulator
default_network_type = "maf" # TODO change default network to integer discrete flows
num_sim = 100 # Number of simulations, number of simulations should increase when number of parameters in the simulation increase
rounds = 2 # Number of rounds to train/simulate

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

def moment_sim_bin(prior: float) -> torch.float32:
    """Uses Moments (cite) to create simulated Site Frequency Spectrums of a demographic history

    Args:
        prior (float): a sampled selection distribution from a prior distribution

    Returns:
        actual_fs: Returns the expected count of alleles, i.e. E[X|gamma] where gamma = prior
        Output of the simulator needs to be float32 based on SBI format 
    """    
    
    moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(10, gamma=prior, theta=100.0))  # returns a masked array
    
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    actual_fs = moment_data.compressed()  
    
    #x = torch.poisson(torch.tensor(actual_fs, device=the_device)).type(torch.float32)
        
    return torch.tensor(actual_fs, device=the_device).type(torch.float32)
  
gamma_prior = torch.distributions.Gamma(torch.tensor([100.0],device=the_device), torch.tensor([10.0],device=the_device))
# Need to create independent distribution so event_shape is 1, see https://bochang.me/blog/posts/pytorch-distributions/ 
# and https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
ind_gamma_prior = torch.distributions.independent.Independent(gamma_prior, 1, validate_args=None)  

# Set up prior and simulator for SBI
prior, num_parameters, prior_returns_numpy = process_prior(ind_gamma_prior)
simulator = process_simulator(moment_sim, prior, prior_returns_numpy)

# First learn posterior
infer_posterior = SNPE(prior, show_progress_bars=True, device='cuda', density_estimator='maf')

#posterior parameters
vi_parameters = dict(q="maf")

proposal = prior
posteriors = []

true_x = moment_sim(prior.sample((1,)).cpu().numpy())

# Train posterior
for i in range(0,rounds):

    theta, x = simulate_for_sbi(simulator, proposal, num_sim)
    liklihood_estimator = infer_posterior.append_simulations(theta, x, ).train(force_first_round_loss=True, training_batch_size=50)
    posterior = infer_posterior.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(true_x).train(max_num_iters=10, quality_control=False )
    accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
    proposal = RestrictedPrior(prior, accept_reject_fn, posterior, sample_with="rejection", device=the_device)


print("Finished Training posterior and prior")

# Set up prior and simulator for SBI

#simulator_bin = process_simulator(moment_sim_bin, bin_proposal, prior_returns_numpy)

# Set up inference scheme for posterior of selection given a specific frequency
'''
bin_proposal, *_ = process_prior(proposal)
inference_bin = SNRE(prior = bin_proposal, device="cuda")

sel_bin_proposal = bin_proposal


sel_bin_posteriors = [[] for _ in range(sample_size)]
true_sel = bin_proposal.sample((1,)).cpu().numpy()
true_fs = moment_sim_bin(true_sel)
'''

# Set up inference scheme for posterior of selection given a specific frequency
bin_proposal, *_ = process_prior(proposal)
simulator_bin = process_simulator(moment_sim_bin, bin_proposal, prior_returns_numpy)
inference_bin = SNRE(prior = bin_proposal, device="cuda")

sel_bin_proposal = bin_proposal

sel_bin_posteriors = [[] for _ in range(sample_size)]
true_sel = bin_proposal.sample((1,)).cpu().numpy()
true_fs = moment_sim_bin(true_sel)

for i in range(0, sample_size, 5):
    
    for _ in range(rounds):
        theta, x = simulate_for_sbi(simulator_bin, sel_bin_proposal, num_sim)
        x2 = x[:, i].unsqueeze(-1)
        print(x2.shape)
        print(theta.shape)
        inference_bin.append_simulations(theta, x2)
        ratio_estimator = inference_bin.train()
        posterior = inference_bin.build_posterior(density_estimator=ratio_estimator)
        sel_bin_posteriors[i].append(posterior)
        #sel_bin_proposal = posterior.set_default_x(true_fs[i].unsqueeze(-1)).train(max_num_iters=40, quality_control=False )
    inference_bin = SNRE(prior = bin_proposal, device="cuda")
    sel_bin_proposal = bin_proposal


'''
for i in range(0, sample_size-1):
    print("**** Doing bin: {} ****".format(i))
    for _ in range(rounds):
        for j in range(0, num_sim):
            theta = sel_bin_proposal.sample((1,))
            x = moment_sim_bin(theta.cpu().numpy())
            #print(x[i].shape)
            #print(x[i].unsqueeze(-1))
            inference_bin.append_simulations(theta, x[i].unsqueeze(-1).unsqueeze(-1), data_device=the_device)
        ratio_estimator = inference_bin.train()
        posterior = inference_bin.build_posterior(density_estimator=ratio_estimator)
        sel_bin_posteriors[i].append(posterior)
        #sel_bin_proposal = posterior.set_default_x(true_fs[i]).train(max_num_iters=40, quality_control=False )
'''
print("Finished calculating posteriors for selection conditioned on frequency")





