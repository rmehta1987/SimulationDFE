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

prior = utils.BoxUniform(low=1 * torch.ones(1), high=100 * torch.ones(1), device='cuda')
pop_dim = 10
posteriors = [None]*(pop_dim-1)
weights = [None]*(pop_dim-1)

# Train SNLE
simulator, prior = prepare_for_sbi(momment_sim, prior)
inferer = SNLE(prior, show_progress_bars=True, device='cuda', density_estimator="maf")
num_sim = 100



# Obtain posterior samples for different number of iid "observed" xos.
num_iid_trials = 100
true_theta = []
true_x = []
# Generate IID samples from the same prior value
for i in range(num_iid_trials):
    theta_o = prior.sample((1,))
    true_theta.append(theta_o)
    x_o, _ = true_data(theta_o)
    true_x.append(x_o)
    
true_x = torch.stack(true_x).cuda()   
    
    

#posterior parameters
vi_parameters = dict(q="maf")

for i in range(1,pop_dim):
    
    theta, x = simulate_for_sbi(simulator, prior, num_sim)
    liklihood_estimator = inferer.append_simulations(theta, x).train()
    a_post = inferer.build_posterior(density_estimator=liklihood_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
    posteriors[i-1] = a_post.set_default_x(true_x).train(max_num_iters=40, quality_control=False )
    
    # Calculate p(i | X) ] = 1/num_sim * (X_i / sum(X_i_to_(pop_dim-1))) where num_sim is the number of simulations
    #step 1 torch.sum(x, axis=1) # calculate sum(X_i_to_(pop_dim-1) for each simulation
    #step 2 x[:,i-1] / (torch.sum(x, axis=1)) # calculate X_i / sum(X_i_to_(pop_dim-1)) for each simulation
    #step 3 calculate 1/num_sim * (X_i / sum(X_i_to_(pop_dim-1)))
    weights[i-1] = torch.mean(x[:,i-1] / (torch.sum(x, axis=1)))


posterior = NeuralPosteriorEnsemble(posteriors, torch.tensor(weights))




# Borrows from poster_ensemble.py to change 
# Number of samples for the selection coefficient
num_samples = torch.tensor((100,pop_dim))
samples = posterior.sample(num_samples)


#True Posterior p(gamma | X, i)

#Step 1, P(X|gamma) * P(gamma)
true_1  = true_x*torch.squeeze(torch.stack(true_theta),1)

#Step 2, P(X|gamma) * P(gamma) * p(i | X) ]
true_post = torch.empty_like(true_x)
for i in range(true_x.shape[0]):
    true_post[i] = true_1[i]*(true_x[i] / torch.sum(true_x[i]))
    
    


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
    

    

