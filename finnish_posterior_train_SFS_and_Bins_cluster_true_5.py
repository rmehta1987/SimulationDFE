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
from sbi.utils.get_nn_models import posterior_nn
import pickle
import os
import seaborn as sns
import datetime
import pandas as pd
import logging
from torch import nn
import torch.nn.functional as F


logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523


# Setup devices and constants
'''
the_device = 'cuda' if torch.cuda.is_available() else 'cpu'
sample_size = 10 # Sample size of population, n
iid = 50  # number of independent samples, but with the same selection coefficeint (i.e. [P(X_1 | gamma_1), P(X_2 | gamma_1), ... , P(X_iid | gamma_1)])
prior_returns_numpy = True # Prior needs to be in numpy format for simulator
default_network_type = "nsf" # TODO change default network to integer discrete flows
num_hidden = 64
number_of_transforms = 3
num_sim = 100 # Number of simulations, number of simulations should increase when number of parameters in the simulation increase
rounds = 2 # Number of rounds to train/simulate
posterior_type = "VI"
num_workers=1
# Large scale simulations
'''

# Setup devices and constants
the_device = 'cuda' if torch.cuda.is_available() else 'cpu'
sample_size = 16734 # Sample size of population, n, (56,885 is the sample size of non-Finnish European overall)
moments_theta = 5288
# number of independent samples, but with the same selection coefficeint (i.e. [P(X_1 | gamma_1), P(X_2 | gamma_1), ... , P(X_iid | gamma_1)])
# only used for SNLE or SNRE
iid = 50  
num_hidden = 256
number_of_transforms = 8
prior_returns_numpy = True # Prior needs to be in numpy format for simulator
default_network_type = "nsf" # TODO change default network to integer discrete flows
num_sim = 100 # Number of simulations, number of simulations should increase when number of parameters in the simulation increase
rounds = 300 # Number of rounds to train/simulate
posterior_type = "VI"
num_workers=1


N0 = 8100 # initial effective pop size
l = [
[2*N0] * 45000,
[2*2000 * np.exp(0.015 * t) for t in range(270)],
[2*1000 * np.exp(0.05 * t) for t in range(87)], # using 5% growth
[(2*1000 * np.exp(0.05 * 87)) * np.exp(0.3 * t) for t in range(1, 13)], # using of 30% growth
]
flat_list = [item for sublist in l for item in sublist]
Nc = np.array(flat_list)/(2*N0) # moments needs pop size as floats scaled by initial effective pop size
NU_FUNC = lambda t: [Nc[int(t*2*N0)]] # creating a function to return *scaled* Ne at each time point/gen

def get_state(post):
    post._optimizer = None
    post.__deepcopy__ = None
    post._q_build_fn = None
    post._q.__deepcopy__ = None

    return post

print("Using device: {}".format(the_device))

def moment_sim(prior: float) -> torch.float32:
    """Uses Moments (cite) to create simulated Site Frequency Spectrums in a constant population size

    Args:
        prior (float): a sampled selection distribution from a prior distribution

    Returns:
        x: The sampled site-frequency spectrum based on moments via Poisson(E[X | gamma]) where gamma = prior
        Output of the simulator needs to be float32 based on SBI format 
    """    
    
    moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(sample_size, gamma=prior, theta=moments_theta))  # returns a masked array
    
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    actual_fs = moment_data.compressed()
    x = torch.poisson(torch.tensor(actual_fs, device=the_device)).type(torch.float32)
        
    return x

def moment_sim_demo(prior: float) -> torch.float32:
    """Uses Moments (cite) to create simulated Site Frequency Spectrums of a demographic history

    Args:
        prior (float): a sampled selection distribution from a prior distribution

    Returns:
        x: The sampled site-frequency spectrum based on moments via Poisson(E[X | gamma]) where gamma = prior
        Output of the simulator needs to be float32 based on SBI format 
    """    
    moment_data = moments.LinearSystem_1D.steady_state_1D(sample_size, gamma=prior, theta=moments_theta)
    moment_data = moments.Spectrum(moment_data)
    moment_data.integrate(NU_FUNC, 2.8, gamma=prior, dt_fac=0.02, theta=moments_theta) # 2.8 is the total number of gens in pop size units
    actual_fs = moment_data.compressed()
    if np.any(actual_fs < 0):
        print("Negative frequency density")
        actual_fs = F.relu(torch.tensor(actual_fs, device=the_device))
        x = torch.poisson(actual_fs).type(torch.float32)
    else:
        x = torch.poisson(torch.tensor(actual_fs, device=the_device)).type(torch.float32)

    return x  


def moment_sim_bin(prior: float) -> torch.float32:
    """Uses Moments (cite) to create simulated Site Frequency Spectrums of a demographic history

    Args:
        prior (float): a sampled sbigmemelection distribution from a prior distribution

    Returns:
        actual_fs: Returns the expected count of alleles, i.e. E[X|gamma] where gamma = prior
        Output of the simulator needs to be float32 based on SBI format 
    """    
    
    moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(sample_size, gamma=prior, theta=moments_theta))  # returns a masked array
    
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    actual_fs = moment_data.compressed()  
    
    #x = torch.poisson(torch.tensor(actual_fs, device=the_device)).type(torch.float32)
        
    return torch.tensor(actual_fs, device=the_device).type(torch.float32)

def get_true_data(a_path: str) -> torch.float32:
    """Gets the true Site-Frequency Spectrum 

    Args:
        path (str): Where the true-SFS is located, must be a text file with separated by columns, the last column represents the Allele count

    Returns:
        Returns the SFS of the true-data set
    """   
    loaded_file = np.loadtxt(a_path)
    ac_count = loaded_file[:,-1].astype(int) # get last column which is the frequency of the alternate/derived allele
    # only get sites > 0
    ac_count = ac_count[np.nonzero(ac_count)]
    thebins = np.arange(1,sample_size+1) # (+1) so that the returned histogram is the same shape as sample_size-1
    # Get the histogram
    sfs, _ = np.histogram(ac_count, bins=thebins)  
    assert sfs.shape[0] == sample_size-1, "Sample Size must be the same dimensions as the Site Frequency Spectrum, SFS shape: {} and sample shape: {}".format(sfs.shape[0], sample_size)

    return torch.tensor(sfs, device=the_device).type(torch.float32)

utils.sbiutils.seed_all_backends(10) # set seed for reproducabilty

# Gamma priors
#gamma_prior = torch.distributions.Gamma(torch.tensor([100.0],device=the_device), torch.tensor([10.0],device=the_device))
# Need to create independent distribution so event_shape is 1, see https://bochang.me/blog/posts/pytorch-distributions/ 
# and https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
#ind_gamma_prior = torch.distributions.independent.Independent(gamma_prior, 1, validate_args=None)  

# Box Uniform priors
box_uniform_prior = utils.BoxUniform(low=-100.0 * torch.ones(1), high=1.0 * torch.ones(1))

# Set up prior and simulator for SBI
#prior, num_parameters, prior_returns_numpy = process_prior(ind_gamma_prior)
prior, num_parameters, prior_returns_numpy = process_prior(box_uniform_prior)

#simulator = process_simulator(moment_sim, prior, prior_returns_numpy)
simulator = process_simulator(moment_sim_demo, prior, prior_returns_numpy)

# First learn posterior
density_estimator_function = posterior_nn(model="nsf", hidden_features=num_hidden, num_transforms=number_of_transforms)
infer_posterior = SNPE(prior, show_progress_bars=True, device=the_device, density_estimator=density_estimator_function)

#posterior parameters
vi_parameters = {"q": "nsf", "parameters": {"num_transforms": 3, "hidden_dims": 32, "skip_connections": True, "nonlinearity": nn.SiLU()}}

proposal = prior
posteriors = []
posteriors2 = []
proposals = []
accept_reject_fns = []

#true_x = moment_sim(prior.sample((1,)).cpu().numpy())
true_x = get_true_data('/project2/jjberg/mehta5/SFSproject/gnomAD/finnish_non_neuro.vcf')
print(true_x.shape)
# Set path for experiments
path = "Experiments/saved_posteriors_finnish_vipost_demo_5000_{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
if not (os.path.isdir(path)):
    try:
        os.mkdir(path)
    except OSError:
        print("Error in making directory")

# Save true data
'''with open("{}/true_x.pkl".format(path), "wb") as handle:
        torch.save(true_x, handle)

# Save initial prior
with open("{}/initial_proposal.pkl".format(path), "wb") as handle:
        temp = prior.sample((10000,1))
        torch.save(temp, handle)
''' 
# Train posterior
print("Starting to Train")
for i in range(0,rounds):

    theta, x = simulate_for_sbi(simulator, proposal, num_sim, num_workers=num_workers)
    liklihood_estimator = infer_posterior.append_simulations(theta, x, ).train(force_first_round_loss=True, training_batch_size=50)
    print("\n ****************************************** Building Posterior for round {} ******************************************.\n".format(i))
    posterior = infer_posterior.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
    #posterior = infer_posterior.build_posterior(density_estimator=liklihood_estimator, sample_with = "rejection")
    posteriors.append(posterior)
    # This proposal is used for Varitaionl inference posteior
    #proposal = posterior.set_default_x(true_x).train(max_num_iters=10, quality_control=False )

    # This proposal is used for Direct posterior inference
    posterior = posterior.set_default_x(true_x).train()
    posteriors2.append(posterior)
    accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
    accept_reject_fns.append(accept_reject_fn)
    proposal = RestrictedPrior(prior, accept_reject_fn, posterior, sample_with="sir", device=the_device)
    proposals.append(proposal)

    ### TODO Create a Plotting UTILS function
    #temp = torch.cat((posteriors[i].sample((1000,)).unsqueeze(1), posteriors2[i].sample((1000,)).unsqueeze(1), proposals[i].sample((1000,)).unsqueeze(1)),dim=1)
    #temp = temp.squeeze()
    #print(temp.shape)
    #df = pd.DataFrame(temp.numpy(), columns=['Posterior', 'Observed Posterior', 'Proposal'])
    #sns.histplot(temp.squeeze(), label=['Posterior', 'Observed Posterior', 'Proposal'], multiple="stack")
    #plt.hist(temp.numpy(), label=['Posterior', 'Observed Posterior', 'Proposal'])
    '''sns.histplot(df, multiple='stack')
    #plt.legend(['Posterior', 'Observed Posterior', 'Proposal'])
    #plt.legend()
    plt.xlabel('Selection (2Ns)')
    plt.ylabel('Counts')
    plt.title('Histogram of Selection Coefficients Sampled from Inferred Distributions of round {}'.format(i))
    plt.tight_layout()
    plt.savefig('{}/histogram_inference_round_{}.png'.format(path, i))
    plt.close()
    
    
    sns.kdeplot(df)

    #plt.legend()
    plt.xlabel('Selection (2Ns)')
    plt.ylabel('Counts')
    plt.title('KDE plot of Selection Coefficients Sampled from Inferred Distributions of round {}'.format(i))
    plt.tight_layout()
    plt.savefig('{}/kde_inference_round_{}.png'.format(path, i))
    plt.close()'''
    #################################

    # Save posters every some rounds
    if i % 5 == 0:
        if posterior_type == "VI":
            # Save posteriors just in case
            path1 = path+"/posterior_round_{}.pkl".format(i)
            path3 = path+"/posterior_observed_round_{}.pkl".format(i)
            with open(path1, "wb") as handle:
                temp = posteriors[i]
                post = get_state(temp)
                torch.save(post, handle)
            with open(path3, "wb") as handle:
                temp2 = posteriors2[i]
                post = get_state(temp2)
                torch.save(post, handle)
        else:
            # Save posteriors just in case
            path1 = path+"/posterior_round_{}.pkl".format(i)
            path3 = path+"/posterior_observed_round_{}.pkl".format(i)
            with open(path1, "wb") as handle:
                torch.save(posteriors[i], handle)
            with open(path3, "wb") as handle:
                torch.save(posteriors2[i], handle)
        print("\n ****************************************** Saved Posterior for round {} ******************************************.\n".format(i))


# Save posteriors and proposals for later use

print("Finished Training posterior and prior")

# used for checking plots 
'''
temp = posteriors[-1].sample((1000,))
sns.kdeplot(temp.squeeze(), label="posteior", color='b')

temp = posteriors2[-1].sample((1000,))
sns.kdeplot(temp.squeeze(), label="observed posteior", color='g')

temp = proposals[-1].sample((1000,))
sns.kdeplot(temp.squeeze(), label="proposal", color='r')
plt.legend()
plt.savefig('test_nsf.png')
'''
### Currently saving objects does not seem to work for all posteriors/samplers (SRI, restricted estimator)

# save Last posterior and observed posterior
if posterior_type=='VI':
    path1 = path+"/posterior_last_round.pkl"
    path2 = path+"/proposal_last_round.pkl"
    path3 = path+"/posterior_observed_last_round.pkl"
    path4 = path+"/accept_function_last_round.pkl"
    with open(path1, "wb") as handle:
        post = get_state(posteriors[-1])
        torch.save(post, handle)
    
    # Can just re-estimate Restricted prior using sampling instead of saving
    # because it cannot save a copy of the get_density_thresholder function
    '''
    try:
        with open(path2, "wb") as handle:
            torch.save(proposals[i], handle)
    except:
        if i == 1:
            print("Cannot save proposal distributions")
        os.remove(path2)
    '''
    with open(path3, "wb") as handle:
        post = get_state(posteriors2[-1])
        torch.save(post, handle)
else:
    # Save posteriors just in case
    path1 = path+"/posterior_last_round.pkl"
    path3 = path+"/posterior_observed_last_round.pkl"
    with open(path1, "wb") as handle:
        torch.save(posteriors[i], handle)
    with open(path3, "wb") as handle:
        torch.save(posteriors2[i], handle)


print("\n ****************************************** Completely finished experiment ****************************************** ")