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

def momment_sim(prior):
    
    moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(10, gamma=prior, theta=100.0))  # returns a masked array
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    actual_fs = moment_data.compressed()  
    x = torch.poisson(torch.tensor(actual_fs))
    return x

def true_data(prior):
    
    if prior.shape[0] > 1:
        moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(10, gamma=prior[0].detach().cpu().numpy(), theta=100.0))
        actual_fs = moment_data.compressed()  
        x = torch.poisson(torch.tensor(actual_fs).repeat(prior.shape[0],1))
    else:
        moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(10, gamma=prior.detach().cpu().numpy(), theta=100.0))
        actual_fs = moment_data.compressed()  
        x = torch.poisson(torch.tensor(actual_fs))
    return x, actual_fs
'''
def a_posterior():
    
    num_iid_trials = 10

    vi_parameters = dict(q="maf")
    posterior = inferer.build_posterior(
    sample_with = "vi",
    vi_method="rKL",
    vi_parameters=vi_parameters
    )
    theta_o = prior.sample((1,))
    x_o = true_data(theta_o.repeat(num_iid_trials,1))
    
    posterior.set_default_x(x_o)
    posterior.train(max_num_iters=40, quality_control=False )
'''


prior = utils.BoxUniform(low=1 * torch.ones(1), high=100 * torch.ones(1), device='cuda')
pop_dim = 10
posteriors = [None]*(pop_dim-1)
weights = [None]*(pop_dim-1)

# Train SNLE
simulator, prior = prepare_for_sbi(momment_sim, prior)
inferer = SNLE(prior, show_progress_bars=True, device='cuda', density_estimator="maf")
num_sim = 10
# theta, x = simulate_for_sbi(simulator, prior, 100)
# inferer.append_simulations(theta, x).train(training_batch_size=10)



# Obtain posterior samples for different number of iid "observed" xos.
num_iid_trials = 10
# Generate IID samples from the same prior value
theta_o = prior.sample((1,))
print("theta for true: {}".format(theta_o))
x_o, actual_fs = true_data(theta_o.repeat(num_iid_trials,1))

#posterior parameters
vi_parameters = dict(q="maf")




'''
# Generate a posterior for every bin (number of features, x.shape[1])
for i in range(1,pop_dim):
    
    theta, x = simulate_for_sbi(simulator, prior, 100)
    bootstrapped_weight = []
    _ = inferer.append_simulations(theta, x).train(training_batch_size=20)
    a_post = inferer.build_posterior(sample_with = "vi", vi_method="rKL", vi_parameters=vi_parameters)
    
    #
    theta_o = prior.sample((1,))
    x_o, _ = true_data(theta_o.repeat(num_iid_trials,1))
    a_post.set_default_x(x_o)
    a_post.train()
    #
        
    
    posteriors[i-1] = a_post
    for a_theta in theta:
        _, fs = true_data(a_theta)
        bootstrapped_weight.append(fs[i-1]/fs.sum())  # the normalizing poisson rate E[X_i|gamma] / [(sum_1_j) E[X_i|gamma])
            
    weights[i-1] = np.mean(bootstrapped_weight)

 ''' 


for i in range(1,pop_dim):
    
    theta, x = simulate_for_sbi(simulator, prior, num_sim)
    liklihood_estimator = inferer.append_simulations(theta, x).train()
    #potential, transform = likelihood_estimator_based_potential(liklihood_estimator, prior, x[0])
    a_post = inferer.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="rKL", vi_parameters=vi_parameters)
    posteriors[i-1] = a_post.set_default_x(x_o).train()
    
    # Calculate p(i | X) ] = 1/num_sim * (X_i / sum(X_i_to_(pop_dim-1))) where num_sim is the number of simulations
    #step 1 torch.sum(x, axis=1) # calculate sum(X_i_to_(pop_dim-1) for each simulation
    #step 2 x[:,i-1] / (torch.sum(x, axis=1)) # calculate X_i / sum(X_i_to_(pop_dim-1)) for each simulation
    #step 3 calculate 1/num_sim * (X_i / sum(X_i_to_(pop_dim-1)))
    weights[i-1] = torch.mean(x[:,i-1] / (torch.sum(x, axis=1)))

    
# train ensemble components
# ensemble_size = pop_dim
# posteriors = Parallel(n_jobs=-1)(
#     delayed(infer)(simulator, prior, "SNLE", num_sim)
#     for i in range(ensemble_size)
# )

# The simplest of posterior using MCMC -- but takes very long
""" for i in range(0,pop_dim-1):
    posteriors[i]=infer(simulator, prior, "SNLE", num_sim)

# create ensemble
posterior = NeuralPosteriorEnsemble(posteriors)
posterior.set_default_x(x_o)
 """

posterior = NeuralPosteriorEnsemble(posteriors, torch.tensor(weights))




# Borrows from poster_ensemble.py to change 
# Number of samples for the selection coefficient
num_samples = torch.tensor((100,pop_dim))

samples = posterior.sample(num_samples)

print(samples)



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
    

    

