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

# Setup devices and constants
the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_size = 10 # Sample size of population, n
iid = 50  # number of independent samples, but with the same selection coefficeint (i.e. [P(X_1 | gamma_1), P(X_2 | gamma_1), ... , P(X_iid | gamma_1)])
prior_returns_numpy = True # Prior needs to be in numpy format for simulator
default_network_type = "maf" # TODO change default network to integer discrete flows
num_sim = 100 # Number of simulations, number of simulations should increase when number of parameters in the simulation increase
rounds = 5 # Number of rounds to train/simulate

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

class TrainedPosterior:
    """This posterior has been trained on a specific observation of the site-frequency spectrum

    Custom prior with user-defined valid .sample and .log_prob methods.
    """

    def __init__(self, a_proposal, device, return_numpy: bool = False):
        self.prior = a_proposal
        self.return_numpy = return_numpy

    def sample(self, sample_shape=torch.Size([])):
        samples = self.prior.sample(sample_shape)
        return samples.cpu().numpy() if self.return_numpy else samples

    def log_prob(self, values):
        log_probs = self.prior.log_prob(values)
        return log_probs


# Set up prior and simulator for SBI

#simulator_bin = process_simulator(moment_sim_bin, bin_proposal, prior_returns_numpy)

# Set up inference scheme for posterior of selection given a specific frequency
proposal = torch.load('restricted.pkl')

inference_bin = SNRE(device='cuda')
bin_proposal = proposal
sel_bin_proposal, *_ = process_prior(TrainedPosterior(bin_proposal, the_device))

vi_parameters = dict(q="maf")

sel_bin_posteriors = [] 
true_fs = moment_sim_bin(bin_proposal.sample((1,)).cpu().numpy())

for i in range(0, sample_size-1):
    for _ in range(rounds):
        for j in range(0, num_sim):
            theta = sel_bin_proposal.sample((1,))
            x = moment_sim_bin(theta.cpu().numpy())
            #print(x[i].shape)
            #print(x[i].unsqueeze(-1))
            inference_bin.append_simulations(theta, x[i].unsqueeze(-1).unsqueeze(-1), data_device=the_device)
        ratio_estimator = inference_bin.train()
        posterior = inference_bin.build_posterior(prior=sel_bin_proposal, density_estimator=ratio_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
        sel_bin_posteriors.append(posterior)
        sel_bin_proposal = posterior.set_default_x(true_fs[i]).train(max_num_iters=40, quality_control=False )