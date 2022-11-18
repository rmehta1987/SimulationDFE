import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, SNLE
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
import numpy as np
import moments
from matplotlib import pyplot as plt
import dill
import os
import seaborn as sns
from joblib import Parallel, delayed
from sbi.inference import infer

# Simulator
def momment_sim(prior):
    
    moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(100, gamma=prior.detach().cpu().numpy(), theta=100.0))  # returns a masked array
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    actual_fs = moment_data.compressed()  
    x = torch.poisson(torch.tensor(actual_fs))
    return x.view(1,-1).type(torch.float32)

def true_data(prior):
    
    if prior.shape[0] > 1:
        moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(100, gamma=prior[0].detach().cpu().numpy(), theta=100.0))
        actual_fs = moment_data.compressed()  
        x = torch.poisson(torch.tensor(actual_fs).repeat(prior.shape[0],1))
    else:
        moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(100, gamma=prior.detach().cpu().numpy(), theta=100.0))
        actual_fs = moment_data.compressed()  
        x = torch.poisson(torch.tensor(actual_fs))
    return x, actual_fs

# Generate Prior
boxprior = utils.BoxUniform(low=1 * torch.ones(1), high=100 * torch.ones(1), device='cuda')
prior = torch.distributions.Gamma(10.0*torch.ones([1.0],device='cuda'), 100*torch.ones(1,device='cuda'))
#prior = torch.distributions.Gamma(torch.tensor([10.0]), torch.tensor([100.0]))
pop_dim = 100
prior = boxprior

# Train SNLE
#simulator, prior = prepare_for_sbi(momment_sim, prior)
inferer = SNLE(prior, show_progress_bars=True, device='cuda', density_estimator="maf")


# Obtain posterior samples for different number of iid "observed" xos.
num_iid_trials = 100

# Obtain posterior samples for different number of iid "observed" xos.
num_sim = 500
num_iid_trials = num_sim # if not num_sim it will not run

# Generate IID samples from the same prior value
theta_o = prior.sample((1,))
print("theta for true: {}".format(theta_o))
true_x, actual_fs = true_data(theta_o.repeat(num_iid_trials,1))
true_x = true_x.cuda()

#posterior parameters
vi_parameters = dict(q="maf")
rounds = 100
'''
posteriors = []
for i in range(0,rounds):
    
    if i == 0:
        theta, x = simulate_for_sbi(momment_sim, prior, num_sim)
        liklihood_estimator = inferer.append_simulations(theta, x).train(training_batch_size=50)
        a_post = inferer.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
        posteriors.append(a_post)
        proposal = a_post.set_default_x(true_x).train(max_num_iters=40, quality_control=False )
    else:
        theta, x = simulate_for_sbi(momment_sim, proposal, num_sim)
        liklihood_estimator = inferer.append_simulations(theta, x).train(training_batch_size=50)
        a_post = inferer.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
        posteriors.append(a_post)
        proposal = a_post.set_default_x(true_x).train(max_num_iters=40, quality_control=False )


'''

from sbi.inference import likelihood_estimator_based_potential
from sbi.inference.posteriors.vi_posterior import VIPosterior

posteriors = []
for i in range(0,rounds):
    
    if i == 0:
        theta, x = simulate_for_sbi(momment_sim, prior, num_sim)
        liklihood_estimator = inferer.append_simulations(theta, x).train(training_batch_size=50)
        potential_fn, theta_transform = likelihood_estimator_based_potential(liklihood_estimator, prior,x_o=true_x[0])
        a_post = VIPosterior(potential_fn=potential_fn,
                        theta_transform=theta_transform,
                        prior=prior, vi_method="IW", q="nsf")
        posteriors.append(a_post)
        proposal = a_post.train(max_num_iters=40, quality_control=False, K=50)
    else:
        print(proposal)
        theta, x = simulate_for_sbi(momment_sim, proposal, num_sim)
        
        liklihood_estimator = inferer.append_simulations(theta, x).train(training_batch_size=50)
        potential_fn, theta_transform = likelihood_estimator_based_potential(liklihood_estimator, proposal,x_o=true_x[0])
        a_post = VIPosterior(potential_fn=potential_fn,
                        theta_transform=theta_transform,
                        prior=prior, vi_method="IW", q="nsf")
        posteriors.append(a_post)
        proposal = a_post.train(max_num_iters=40, quality_control=False, K=50)


'''
# Plot them in one pairplot as contours (obtained via KDE on the samples).
fig, ax = analysis.pairplot(
    samples,
    points=theta_o.cpu(),
    diag="kde",
    upper="contour",
    kde_diag=dict(bins=100),
    kde_offdiag=dict(bins=50),
    contour_offdiag=dict(levels=[0.95]),
    points_colors=["red"],
    points_offdiag=dict(marker="*", markersize=10),
)
plt.savefig("testSNLE3.png")

# sample gammes from the posterior
post_gammas = posterior.sample(sample_shape=(100,))
post_x2 = []
for g in post_gammas:
    x2 = true_data(g)
    post_x2.append(torch.reshape(x2,(1,x2.shape[0])))
'''
    

    

