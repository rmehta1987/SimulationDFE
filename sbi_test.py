import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, SNLE
import numpy as np
import moments
from matplotlib import pyplot as plt
import dill
import os
import seaborn as sns

def simulator(prior):
    
    moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(100, gamma=prior, theta=100.0))  # returns a masked array
    # masked arrays are objects and data is accessed through .data attribute or valid data through .compressed()
    actual_fs = moment_data.compressed()  
    x = torch.poisson(torch.tensor(actual_fs))
    return x

def true_data(prior):
    
    if prior.shape[0] > 1:
        moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(100, gamma=prior[0].detach().cpu().numpy(), theta=100.0))
        actual_fs = moment_data.compressed()  
        x = torch.poisson(torch.tensor(actual_fs).repeat(prior.shape[0],1))
    else:
        moment_data =  moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(100, gamma=prior.detach().cpu().numpy(), theta=100.0))
        actual_fs = moment_data.compressed()  
        x = torch.poisson(torch.tensor(actual_fs))
    return x

prior = utils.BoxUniform(low=1 * torch.ones(1), high=100 * torch.ones(1), device='cuda')

'''
# Train SNPE
simulator, prior = prepare_for_sbi(simulator, prior)
theta, x = simulate_for_sbi(simulator, prior, num_simulations=500)
inference = SNPE(prior)
density_estimator = inference.append_simulations(theta, x).train()
posterior = inference.build_posterior(density_estimator)
true_gamma = prior.sample()
x_o = true_data(true_gamma)
posterior_samples = posterior.sample((10000,), x=x_o)

fig, axes = analysis.pairplot(
    posterior_samples,
    limits=[[0.5, 80], [1e-4, 15.0]],
    ticks=[[0.5, 80], [1e-4, 15.0]],
    figsize=(5, 5),
    points=true_gamma,
    points_offdiag={"markersize": 6},
    points_colors="r",
)

plt.savefig("testSNPE.png")'''


'''
if os.path.exists('inference.pkl'):
    with open("inference.pkl", "rb") as handle:
        inferer = dill.load(handle)
        x = dill.load(open('x.pkl', "rb"))
        theta = dill.load(open('theta.pkl', "rb"))
    print("Loaded pre-trainied density estimator")
else:
    # Train SNLE
    simulator, prior = prepare_for_sbi(simulator, prior)
    inferer = SNLE(prior, show_progress_bars=True, device='cuda', density_estimator="maf")
    theta, x = simulate_for_sbi(simulator, prior, 100)
    inferer.append_simulations(theta, x).train(training_batch_size=10)
    with open("inference.pkl", "wb") as handle:
        dill.dump(inferer, handle)
        dill.dump(theta, file=open('theta.pkl', "wb"))
        dill.dump(x, file=open('x.pkl', "wb"))
    print("Created and saved a trainied density estimator")
'''

 # Train SNLE
simulator, prior = prepare_for_sbi(simulator, prior)
inferer = SNLE(prior, show_progress_bars=True, device='cuda', density_estimator="maf")
theta, x = simulate_for_sbi(simulator, prior, 100)
inferer.append_simulations(theta, x).train(training_batch_size=10)

print(" Finished training SNLE")

# Obtain posterior samples for different number of iid xos.
samples = []
num_samples = 5000
num_iid_trials = 10

# Generate IID samples from the same prior value
theta_o = prior.sample((1,))
x_o = true_data(theta_o.repeat(num_iid_trials,1))

print("Finished generating IID samples from the same gamma")

# mcmc_parameters = dict(
#     num_chains=5,
#     thin=10,
#     warmup_steps=50,
#     init_strategy="proposal",
# )
# mcmc_method = "slice_np_vectorized"

'''
posterior = inferer.build_posterior(
    mcmc_method=mcmc_method,
    mcmc_parameters=mcmc_parameters,
)
'''

vi_parameters = dict(q="maf")
posterior = inferer.build_posterior(
    sample_with = "vi",
    vi_method="rKL",
    vi_parameters=vi_parameters
)

#posterior.set_default_x()
#posterior.set_default_x(torch.tensor(np.zeros((x.shape[1],)).astype(np.float32)))_
posterior.set_default_x(x_o)
posterior.train(max_num_iters=40, quality_control=False )
for xo in x_o:
    samples.append(posterior.sample(sample_shape=(num_samples,)).cpu())
    
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
    
# Plot them in one pairplot as contours (obtained via KDE on the samples).
fig, ax = analysis.pairplot(
    post_x2,
    points=x,
    diag="kde",
    upper="contour",
    kde_diag=dict(bins=100),
    kde_offdiag=dict(bins=50),
    contour_offdiag=dict(levels=[0.95]),
    points_colors=["red"],
    points_offdiag=dict(marker="*", markersize=10),
)
plt.savefig("posttest_SNLE.png")
'''