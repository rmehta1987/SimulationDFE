
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

logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523



class SummaryNet(nn.Module):
    def __init__(self, sample_size, block_sizes, dropout_rate=0.0):
        super().__init__()
        self.sample_size = sample_size # For monarch this needs to be divisible by the block size
        self.block_size = block_sizes
        self.linear4 = MonarchLinear(sample_size, int(sample_size / 10), nblocks=self.block_size[0]) # 11171
        self.linear5 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 50) , nblocks=self.block_size[1]) # 2234.2
        self.linear6 = MonarchLinear(int(self.sample_size / 50), int(self.sample_size / 100), nblocks=self.block_size[2]) # 1117.1

        self.model = nn.Sequential(self.linear4, nn.Dropout(dropout_rate), nn.SiLU(inplace=True),
                                   self.linear5, nn.Dropout(dropout_rate), nn.SiLU(inplace=True),
                                   self.linear6) 
    def forward(self, x):
        
        x=self.model(x)
        return x

    def forward(self, x):
        
        x=self.model(x)
        return x

the_device='cuda:0'


prior = utils.BoxUniform(low=-0.990 * torch.ones(1, device=the_device), high=-1e-8*torch.ones(1,device=the_device),device=the_device)

saved_path='Experiments/saved_posteriors_nfe_infer_lof_selection_bins_2023-03-07_17-11/'

lsdirs = os.listdir(saved_path)
obs_dict = dict()
post_dict = dict()

for round, a_file in enumerate(lsdirs):
    if 'observed' in a_file:
        #round_num = int(re.search("\d+", a_file)[0])
        post_obs = torch.load(f'{saved_path}/{a_file}')
        samples = post_obs.sample((1000,),method='slice_np_vectorized', num_workers=6)
        obs_dict[round] = samples.cpu().squeeze().numpy()
        del post_obs
    else:
        #round_num = int(re.search("\d+", a_file)[0])
        post = torch.load(f'{saved_path}/{a_file}')
        samples = post.sample((100,))
        post_dict[round] = samples.cpu().squeeze().numpy()
        del post

#post_dict['intial'] = prior.sample((100000,)).cpu().squeeze().numpy()
#obs_dict['intial'] = prior.sample((100000,)).cpu().squeeze().numpy()

postdf = pd.DataFrame.from_dict(post_dict)
obsdf = pd.DataFrame.from_dict(obs_dict)


bins = [0, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]



sns.kdeplot(obsdf, label=obsdf.columns)

plt.xlabel('Unscaled Selection (|s|)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof_monarch_bins.png')
plt.close()


sns.kdeplot(np.log(np.abs(obsdf)), label=obsdf.columns)
#plt.legend(['Learned Prior', 'Proposal'])
#plt.legend()
plt.xlabel('Unscaled Selection (log{|s|))')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof_monarch_bins_log.png')
plt.close()

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
plt.savefig('para_lof_log_30.png')
plt.close()
'''


'''
obs90 = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_lof_selection_nvsl_2023-03-02_07-27/posterior_observed_round_30.pkl')
post90 = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_lof_selection_nvsl_2023-03-02_07-27/posterior_round_30.pkl')

accept_reject_fn = get_density_thresholder(obs90, quantile=1e-6)
proposal = RestrictedPrior(prior, accept_reject_fn, post90, sample_with="sir", device=the_device)

dfe = proposal.sample((100000,))
temp3 = -1*torch.cat((dfe, post90.sample((100000,)), prior.sample((100000,))),dim=1)
df3 = pd.DataFrame(temp3.cpu().numpy(), columns=['Restricted prior', 'Training Round 30', 'Initial Propsal'])
temp4 = torch.log10(torch.abs(temp3.squeeze()))

sns.kdeplot(df3, label=['Restricted Round 90', 'Training Round 30', 'Initial Proposal'])
plt.xlabel('Unscaled Selection (|s})')
plt.ylabel('Density')
plt.savefig('para_monarch_dfe_30.png')
plt.close()
temp4 = torch.log10(torch.abs(temp3.squeeze()))
df4 = pd.DataFrame(temp4.cpu().numpy(), columns=['DFE Round 90', 'Training Round 90', 'Initial Propsal'])

sns.kdeplot(df4, label=['DFE Round 90', 'Training Round 30', 'Initial Proposal'])
plt.xlabel('Unscaled Selection (log(|s|)})')
plt.ylabel('Density')
plt.savefig('para_dfe_monarch_log_30.png')
'''