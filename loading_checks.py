
import torch
from torch import nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi, SNLE, MNLE, SNRE, SNRE_A
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from sbi.utils import BoxUniform
from sbi.utils import MultipleIndependent
from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding, FCEmbedding
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi.utils import get_density_thresholder, RestrictedPrior
from sbi.utils.get_nn_models import posterior_nn
import numpy as np
import moments
from matplotlib import pyplot as plt
import pickle
import os
import seaborn as sns
import datetime
import pandas as pd
import logging
import atexit
import torch.nn.functional as F
import subprocess
import sparselinear as sl
from sortedcontainers import SortedDict
from scipy.spatial import KDTree
import os
import re
from monarch_linear import MonarchLinear
import pdb
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523
from collections import defaultdict
from sbi.analysis import pairplot


the_device='cuda:0'
sample_size = 85
bins = [0, 1e-5, 1e-4, 1e-3, 1e-2]

prior = utils.BoxUniform(low=-0.2 * torch.ones(1, device=the_device), high=-1e-5*torch.ones(1,device=the_device),device=the_device)


class SummaryNet(nn.Module):
    def __init__(self, sample_size, block_sizes, dropout_rate=0.0):
        super().__init__()
        self.sample_size = sample_size # For monarch this needs to be divisible by the block size
        self.block_size = block_sizes
        self.linear4 = MonarchLinear(sample_size, int(sample_size / 10), nblocks=self.block_size[0]) # 11171
        self.linear5 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 10) , nblocks=self.block_size[1]) # 11171
        self.linear6 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 10), nblocks=self.block_size[2]) # 11171

        self.model = nn.Sequential(self.linear4, nn.Dropout(dropout_rate), nn.GELU(),
                                   self.linear5, nn.Dropout(dropout_rate), nn.GELU(),
                                   self.linear6) 
    def forward(self, x):
        
        x=self.model(x)
        return x

    
class SummaryNet5(nn.Module):
    # in_features * out_features <= 10**8:
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.linear4 = sl.SparseLinear(int(self.sample_size), int(self.sample_size / 125), dynamic=True) # 893.68
        self.bn4 = nn.LayerNorm(int(self.sample_size / 125))
        self.linear5 = sl.SparseLinear(int(self.sample_size / 125), int(self.sample_size / 175), dynamic=True) # 638.34
        self.bn5 = nn.LayerNorm(int(self.sample_size / 175))
        self.linear6 = sl.SparseLinear(int(self.sample_size / 175), int(self.sample_size / 200), dynamic=True) # 585.5
        self.bn6 = nn.LayerNorm(int(self.sample_size / 200))
        self.linear7 = sl.SparseLinear(int(self.sample_size / 200), int(self.sample_size / 400), dynamic=True) # 279.5
        self.bn7 = nn.LayerNorm(int(self.sample_size / 400))
        self.linear8 = sl.SparseLinear(int(self.sample_size / 400), int(self.sample_size / 400), dynamic=True) # 279.25


        self.model = nn.Sequential(self.linear4, self.bn4, nn.SiLU(inplace=True),
                                   self.linear5, self.bn5, nn.SiLU(inplace=True),
                                   self.linear6, self.bn6, nn.SiLU(inplace=True),
                                   self.linear7, self.bn7, nn.SiLU(inplace=True),
                                   self.linear8,)
                                   
                                   

    def forward(self, x):
        
        x=self.model(x)
        return x

class SummaryNet7(nn.Module):
    # in_features * out_features <= 10**8:
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.linear4 = sl.SparseLinear(int(self.sample_size), int(self.sample_size / 125), dynamic=True) # 893.68
        self.bn4 = nn.LayerNorm(int(self.sample_size / 125))
        self.linear5 = sl.SparseLinear(int(self.sample_size / 125), int(self.sample_size / 175), dynamic=True) # 638.34
        self.bn5 = nn.LayerNorm(int(self.sample_size / 175))
        self.linear6 = sl.SparseLinear(int(self.sample_size / 175), int(self.sample_size / 200), dynamic=True) # 585.5
        self.bn6 = nn.LayerNorm(int(self.sample_size / 200))
        self.linear7 = sl.SparseLinear(int(self.sample_size / 200), int(self.sample_size / 400), dynamic=True) # 279.5
        self.bn7 = nn.LayerNorm(int(self.sample_size / 400))
        self.linear8 = sl.SparseLinear(int(self.sample_size / 400), int(self.sample_size / 400), dynamic=True) # 279.25


        self.model = nn.Sequential(self.linear4, self.bn4, nn.SiLU(inplace=True),
                                   self.linear5, self.bn5, nn.SiLU(inplace=True),
                                   self.linear6, self.bn6, nn.SiLU(inplace=True),
                                   self.linear7, self.bn7, nn.SiLU(inplace=True),
                                   self.linear8,)
                                   
                                   

    def forward(self, x):
        
        x=self.model(x)
        return x

def ImportanceSamplingEstimator(sample, threshold, target=None, num_particles=None):
    """_summary_

    Args:
        sample (_type_): _description_
        target (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_
    """    
    cdftest = target.q.transforms
    low_samples = sample - .0025
    high_samples = sample + .0025 
    num_particles = 1000 if num_particles is None else num_particles
    if target is not None:
        with torch.no_grad():
            for transform in cdftest[::-1]:
                value = transform.inv(high_samples)
            if target.q._validate_args:
                target.q.base_dist._validate_sample(value)
            value = target.q.base_dist.base_dist.cdf(value)
            #value = target.q._monotonize_cdf(value)
        with torch.no_grad():
            for transform in cdftest[::-1]:
                value2 = transform.inv(low_samples)
            if target.q._validate_args:
                target.q.base_dist._validate_sample(value2)
            value2 = target.q.base_dist.base_dist.cdf(value2)
            #value2 = target.q.base_dist._monotonize_cdf(value2)
        
        return value - value2
    else:
        sample_low = sample-threshold
        sample_high = sample+threshold
        proposal = torch.distributions.uniform.Uniform(sample_low, sample_high)
        prop_samps = proposal.sample((num_particles,))
        target_logprobs = target.log_prob(prop_samps)
        proposal_logprobs = proposal.log_prob(prop_samps)
        log_importance_weights = target_logprobs - proposal_logprobs

    ret = torch.sum(torch.exp(log_importance_weights))/num_particles


    return ret 

def generate_moments_sim_data(prior: float) -> torch.float32:
    
    global sample_size
    opt_params = [2.21531687, 5.29769918, 0.55450117, 0.04088086]
    theta_mis = 15583.437265450002
    theta_lof = 1164.3148344084038
    rerun = True
    ns_sim = 100
    h=0.5
    projected_sample_size = sample_size*2
    s_prior, weights = prior[:6], prior[6:]
    #s_prior, weights = prior[:5], prior[5:]
    fs_aggregate = None
    gammas = s_prior.cpu().numpy().squeeze()
    weights = weights.cpu().numpy().squeeze()
    for j, (gamma, weight) in enumerate(zip(gammas, weights)):
        while rerun:
            ns_sim = 2 * ns_sim
            fs = moments.LinearSystem_1D.steady_state_1D(ns_sim, gamma=gamma, h=h)
            fs = moments.Spectrum(fs)
            fs.integrate([opt_params[0]], opt_params[2], gamma=gamma, h=h)
            nu_func = lambda t: [opt_params[0] * np.exp(
                np.log(opt_params[1] / opt_params[0]) * t / opt_params[3])]
            fs.integrate(nu_func, opt_params[3], gamma=gamma, h=h)
            if abs(np.max(fs)) > 10 or np.any(np.isnan(fs)):
                # large gamma-values can require large sample sizes for stability
                rerun = True
            else:
                rerun = False
        if j == 0:
            fs_aggregate = fs.project([projected_sample_size]).compressed()*theta_mis * weight
        else:
            fs_aggregate += fs.project([projected_sample_size]).compressed()*theta_mis * weight

    fs_aggregate = torch.tensor(fs_aggregate).type(torch.float32) 
    return fs_aggregate



embedding_net = SummaryNet(sample_size*2-2, [32, 32, 32]).to(the_device)

saved_path='Experiments/saved_posteriors_msl_nsf2023-03-29_21-29'

'''

lsdirs = os.listdir(saved_path)
obs_dict = defaultdict()
post_dict = defaultdict()

for i, a_file in enumerate(lsdirs):
    if 'observed' in a_file and 'last' not in a_file:
        round_num = int(re.search("\d+", a_file)[0])
        post_obs = torch.load(f'{saved_path}/{a_file}')
        obs_samples = post_obs.sample((100000,))/2.0
        obs_dict[round_num] = {}
        for j in range(0, obs_samples.shape[1]):
            obs_dict[round_num][j] = obs_samples[:,j].cpu().squeeze().numpy()

        #if i < len(lsdirs)-1:
        #    del post_obs
    elif 'posterior' in a_file and 'observed' not in a_file and 'last' not in a_file:
        print(a_file)
        round_num = int(re.search("\d+", a_file)[0])
        post = torch.load(f'{saved_path}/{a_file}')
        post_samples = post.sample((100000,))/2.0
        post_dict[round_num] = {}
        for j in range(0, post_samples.shape[1]):
            post_dict[round_num][j] = post_samples[:,j].cpu().squeeze().numpy()

        #if i < len(lsdirs)-1:
        #    del post

#post_obs = torch.load(f'{saved_path}/{lsdirs[-1]}')

#post_dict['intial'] = prior.sample((100000,)).cpu().squeeze().numpy()
#obs_dict['intial'] = prior.sample((100000,)).cpu().squeeze().numpy()
#postdf = pd.DataFrame.from_dict(post_dict)
#obsdf = pd.DataFrame.from_dict(obs_dict)

fig, ax = pairplot(samples=post_samples.cpu(), upper=["kde"], diag=["hist"])
plt.savefig('post_pair_plot.png')

fig, ax = pairplot(samples=obs_samples.cpu(), upper=["kde"], diag=["hist"])
plt.savefig('observed_pair_plot.png')
'''

# Create a posterior predicitive check using lat rounds
last_posterior = torch.load('Experiments/saved_posteriors_msl_nsf2023-03-29_21-29/posterior_observed_round_35.pkl')
true_x = np.load('emperical_missense_sfs_msl.npy')
predicted_fs=[]
predicted_fs2=[]

obs_samples = last_posterior.sample((3,))/2.0
for obs_sample in obs_samples:
    probs = ImportanceSamplingEstimator(obs_sample, .0025, last_posterior)
    weighted_probs = probs/torch.sum(probs)
    fs = generate_moments_sim_data(torch.cat((obs_sample, weighted_probs), dim=0))
    predicted_fs.append(fs.unsqueeze(0).cpu().numpy())
    predicted_fs2.append(np.log(fs[1:169:20].unsqueeze(0).cpu().numpy()))

predicted_fs2 = np.asarray(predicted_fs2).squeeze(1)
smaller_true_x = np.log(true_x[1:169:20])
_ = pairplot(
    samples=predicted_fs2,
    points=smaller_true_x,
    points_colors="red",
    figsize=(8, 8),
    upper="scatter",
    scatter_offdiag=dict(marker=".", s=5),
    points_offdiag=dict(marker="+", markersize=20),)
    #labels=[rf"$x_{d}$" for d in range(1,obs_samples.shape)])
plt.savefig('ppc_check.png')

'''
sns.kdeplot(obsdf, label=obsdf.columns)

plt.xlabel('Unscaled Selection (|s|)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('absolute_unscaled_selection.png')
plt.close()


sns.kdeplot(np.log(np.abs(obsdf)), label=obsdf.columns)
#plt.legend(['Learned Prior', 'Proposal'])
#plt.legend()
plt.xlabel('Unscaled Selection (log{|s|))')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('msl_log_missense.png')
plt.close()

'''
'''
#plt.hist(np.log10(temp[:,0].cpu().numpy()),bins=[-5, -4, -3, -2, -1, 0], edgecolor='black', linewidth=1.2, histtype='bar')
#plt.hist(np.log10(temp[:,1].cpu().numpy()),bins=[-5, -4, -3, -2, -1, 0], edgecolor='black', linewidth=1.2, histtype='bar')#plt.legend(['Learned Prior', 'Proposal'])
#plt.hist(np.log10(temp[:,2].cpu().numpy()),bins=[-5, -4, -3, -2, -1, 0], edgecolor='black', linewidth=1.2, histtype='bar')#plt.legend(['Learned Prior', 'Proposal'])

plt.hist(temp2.cpu().numpy(), bins=[-5, -4, -3, -2, -1, 0], linewidth=1.2, histtype='step', stacked=True, fill=False)


plt.xlabel('Unscaled Selection (|s})')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof_hist2_30.png')
plt.close()

plt.hist(temp2.cpu().numpy(), bins=[-5, -4, -3, -2, -1, 0], linewidth=1.2, histtype='bar', stacked=False, fill=True)
plt.xlabel('Unscaled Selection (|s|)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof_hist3_30.png')
plt.close()


sns.kdeplot(df2, label=thecolumns)
#plt.legend(['Learned Prior', 'Proposal'])
#plt.legend()
plt.xlabel('Unscaled Selection (log{|s|))')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('msl_log_missense.png')
plt.close()
'''

'''
#obs90 = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_lof_selection_nvsl_2023-03-02_07-27/posterior_observed_round_30.pkl')
#post90 = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_lof_selection_nvsl_2023-03-02_07-27/posterior_round_30.pkl')


#max_round = max(list(postdf.columns))

#post_obs = postdf[max_round]

accept_reject_fn = get_density_thresholder(post_obs, quantile=1e-5)
proposal = RestrictedPrior(prior, accept_reject_fn, post_obs, sample_with="sir", device=the_device)
proposal2 = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection", device=the_device)

dfe = proposal.sample((100000,))/2.0
log_prob_dfe = proposal2.log_prob(dfe*2.0)
temp3 = -1*torch.cat((dfe, post_obs.sample((100000,))/2.0, prior.sample((100000,))),dim=1)
df3 = pd.DataFrame(temp3.cpu().numpy(), columns=['Restricted prior', 'Training Round 30', 'Initial Propsal'])
temp4 = torch.log10(torch.abs(temp3.squeeze()))

sns.kdeplot(df3, label=['Restricted Round 90', 'Training Round 30', 'Initial Proposal'])
plt.xlabel('Unscaled Selection (|s})')
plt.ylabel('Density')
plt.savefig('para_dfe_misenese_monarch_nsf_lof_30.png')
plt.close()
temp4 = torch.log10(torch.abs(temp3.squeeze()))
df4 = pd.DataFrame(temp4.cpu().numpy(), columns=['DFE Round 90', 'Training Round 90', 'Initial Propsal'])

sns.kdeplot(df4, label=['DFE Round 90', 'Training Round 30', 'Initial Proposal'])
plt.xlabel('Unscaled Selection (log(|s|)})')
plt.ylabel('Density')
plt.savefig('para_dfe_monarch_log_lof_nsf_30.png')


dfe2 = proposal.sample((500000,))/2.0
dfe2 = np.abs(dfe2.squeeze().cpu().numpy())

for s0, s1 in zip(bins[:-1], bins[1:]):
    the_dat=np.extract((s0 <= dfe2) & (dfe2 < s1), dfe2)
    prop = the_dat.shape[0]/500000.0
    print(f"{s0} <= s < {s1}: {prop:.7f}")
    if s1 == bins[-1]:
        the_dat=np.extract(dfe2 > s1, dfe2)
        prop = the_dat.shape[0]/500000.0
        print(f"s > {s1}: {prop:.7f}")



'''

