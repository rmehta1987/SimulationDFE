import moments
import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi.utils import MultipleIndependent
from torch.distributions import Beta, Binomial, Gamma
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.utils.user_input_checks import process_prior, process_simulator
from torch import nn


'''
sampled_demes = ["YRI"]
sample_sizes = [40]
gamma = -20.0
h = 0.1
ooa_model = '/home/rahul/PopGen/SimulationSFS/gutenkunst_ooa.yaml'

fs_yri_sel = moments.Spectrum.from_demes(ooa_model, sampled_demes=sampled_demes, sample_sizes=sample_sizes, gamma=gamma, h=h)
fs_yri = moments.Spectrum.from_demes(ooa_model, sampled_demes=sampled_demes, sample_sizes=sample_sizes)


fig = plt.figure()
ax = plt.subplot(111)
ax.semilogy(fs_yri, "-o", ms=6, lw=1, mfc="w", label="Neutral");
ax.semilogy(fs_yri_sel, "-o", ms=3, lw=1,
    label=f"Selected, $\gamma={gamma}$, $h={h}$");
ax.set_ylabel("Density");
ax.set_xlabel("Derived allele count");
ax.legend()
plt.show()
'''
sample_size = 60
the_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def moment_sim_demo(prior):
    """Uses Moments (cite) to create simulated Site Frequency Spectrums of a demographic history

    Args:
        prior (float): a sampled selection distribution from a prior distribution

    Returns:
        x: The sampled site-frequency spectrum based on moments via Poisson(E[X | gamma]) where gamma = prior
        Output of the simulator needs to be float32 based on SBI format 
    """    
    nuB = prior[0]
    nuF = prior[1]
    etime = prior[2]
    theta = prior[4]
    moment_data =  moments.Demographics1D.bottlegrowth([nuB, nuF, etime], [sample_size]) 
    moment_data = theta * moment_data
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    data = moment_data.sample()
        
    return torch.tensor((data), device=the_device).type(torch.float32)

# nuB = 0.2
# nuF = 3.0
# etime = 0.4
# theta = 2000.0
prior = MultipleIndependent(
    [
        torch.distributions.Uniform(torch.tensor([0.1]), torch.tensor([0.5])),
        torch.distributions.Uniform(torch.tensor([1.0]), torch.tensor([5.0])),
        torch.distributions.Uniform(torch.tensor([1.0]), torch.tensor([5.0])),
        torch.distributions.Uniform(torch.tensor([1000.0]), torch.tensor([3000.0]))
    ],
    validate_args=False,
)



torch.manual_seed(42)
num_trials = 10
num_samples = 1000
the_device = 'cuda' if torch.cuda.is_available() else 'cpu'
iid = 50  # number of independent samples, but with the same selection coefficeint (i.e. [P(X_1 | gamma_1), P(X_2 | gamma_1), ... , P(X_iid | gamma_1)])
prior_returns_numpy = True # Prior needs to be in numpy format for simulator
default_network_type = "nsf" # TODO change default network to integer discrete flows
num_hidden = 64
number_of_transforms = 3
num_sim = 100 # Number of simulations, number of simulations should increase when number of parameters in the simulation increase
rounds = 2 # Number of rounds to train/simulate
posterior_type = "VI"

# First learn posterior
density_estimator_function = posterior_nn(model="nsf", hidden_features=num_hidden, num_transforms=number_of_transforms)
infer_posterior = SNPE(prior, show_progress_bars=True, device=the_device, density_estimator=density_estimator_function)


theta_o = prior.sample((1,))
#x_o = moment_sim(theta_o.repeat(num_trials, 1))
x_o = moment_sim_demo(theta_o)
simulator = process_simulator(moment_sim_demo, prior, True)
posteriors = []
posteriors2 = []
for i in range(0,rounds):

    theta, x = simulate_for_sbi(simulator, proposal, num_sim)
    liklihood_estimator = infer_posterior.append_simulations(theta, x)
    liklihood_estimator = liklihood_estimator.train(training_batch_size=50, discard_prior_samples=True)
    print("\n ****************************************** Building Posterior for round {} ******************************************.\n".format(i))

    posterior = infer_posterior.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
    posteriors.append(posterior)
    posterior = posterior.set_default_x(true_x).train()

    posteriors2.append(posterior)

    
   