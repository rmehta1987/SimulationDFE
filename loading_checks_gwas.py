
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

class SummaryNet(nn.Module):
    def __init__(self, sample_size, block_sizes, dropout_rate=0.0):
        super().__init__()
        self.sample_size = sample_size # For monarch this needs to be divisible by the block size
        self.block_size = block_sizes
        self.linear4 = MonarchLinear(self.sample_size, int(self.sample_size / 10), nblocks=self.block_size[0]) # size: [sample_size-cut_freq, ~72077]
        self.linear5 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 10) , nblocks=self.block_size[1]) # [~72077, ~72077]
        self.linear6 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 10), nblocks=self.block_size[2]) # [~72077, ~72077]
        #self.linear7 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 50), nblocks=self.block_size[3]) # [~72077, ~72077]
        #self.linear8 = MonarchLinear(int(self.sample_size / 50), int(self.sample_size / 50), nblocks=self.block_size[4]) # [~14401, ~14401]
        #self.linear9 = MonarchLinear(int(self.sample_size / 50), int(self.sample_size / 50), nblocks=self.block_size[4]) # [~14401, ~14401]

        self.model = nn.Sequential(self.linear4, nn.Dropout(dropout_rate), nn.LeakyReLU(),
                                   self.linear5, nn.Dropout(dropout_rate), nn.LeakyReLU(),
                                   self.linear6,)
                                   #self.linear7, nn.Dropout(dropout_rate), nn.LeakyReLU(),
                                   #self.linear8, nn.Dropout(dropout_rate), nn.LeakyReLU(),
                                   #self.linear9)
    def forward(self, x):

        x=self.model(x)
        return x

ind_prior = MultipleIndependent(
    [
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param2, high=high_param),
    ],
    validate_args=False,)
    # Set up prior and simulator for SBI
prior, num_parameters, prior_returns_numpy = process_prior(ind_prior)
proposal = prior


# Create a posterior predicitive check using lat rounds
last_posterior = torch.load('nfe_restriction_classifier_gwas_embedding_round3_final.pkl')
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

