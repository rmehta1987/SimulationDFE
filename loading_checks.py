
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

logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523

class SummaryNet(nn.Module):
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.linear1 = sl.SparseLinear(self.sample_size, int(self.sample_size / 200)) # 585
        self.bn1 = nn.LazyBatchNorm1d(int(self.sample_size / 200))
        self.linear2 = sl.SparseLinear(int(self.sample_size / 200), int(self.sample_size / 200)) # 585
        self.linear3 = sl.SparseLinear(int(self.sample_size / 200), int(self.sample_size / 400)) # 283
        self.bn2 = nn.LazyBatchNorm1d(int(self.sample_size / 400))
        self.linear4 = sl.SparseLinear(int(self.sample_size / 400), int(self.sample_size / 600)) # 195
        self.bn3 = nn.LazyBatchNorm1d(int(self.sample_size / 600))
        self.linear5 = sl.SparseLinear(int(self.sample_size / 600), int(self.sample_size / 600)) # 195

        self.model = nn.Sequential(self.linear1, self.bn1, nn.SiLU(), self.linear2, nn.SiLU(), self.linear3, self.bn2, nn.SiLU(), self.linear4, self.bn3, nn.SiLU(), self.linear5)

    def forward(self, x):
        
        x=self.model(x)
        return x
    
saved_path='saved_posteriors_nfe_infer_missense_selection_1_2023-02-22_13-56'
obs20 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_20.pkl')
obs15 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_15.pkl')
obs10 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_10.pkl') 
obs5 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_5.pkl')
obs0 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_0.pkl')

sampleobs20 = obs20.sample((100000,))
sampleobs15 = obs15.sample((100000,))
sampleobs10 = obs10.sample((100000,))
sampleobs5 = obs5.sample((100000,))
sampleobs0 = obs0.sample((100000,))
bins = [0, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]

temp = -1*torch.cat((sampleobs20, sampleobs15, sampleobs10, sampleobs5, sampleobs0),dim=1)
temp2 = torch.log10(torch.abs(temp.squeeze()))
print(temp.shape)
thecolumns = ['Training Round {}'.format(i*5) for i in range(0,temp.shape[1])]
thecolumns.reverse()
print(thecolumns)
df = pd.DataFrame(temp.cpu().numpy(), columns=thecolumns)
df2 = pd.DataFrame(temp2.cpu().numpy(), columns=thecolumns)
sns.kdeplot(df, label=thecolumns)
#plt.legend(['Learned Prior', 'Proposal'])
#plt.legend()
plt.xlabel('Unscaled Selection (s)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof.png')
plt.close()

#plt.hist(np.log10(temp[:,0].cpu().numpy()),bins=[-5, -4, -3, -2, -1, 0], edgecolor='black', linewidth=1.2, histtype='bar')
#plt.hist(np.log10(temp[:,1].cpu().numpy()),bins=[-5, -4, -3, -2, -1, 0], edgecolor='black', linewidth=1.2, histtype='bar')#plt.legend(['Learned Prior', 'Proposal'])
#plt.hist(np.log10(temp[:,2].cpu().numpy()),bins=[-5, -4, -3, -2, -1, 0], edgecolor='black', linewidth=1.2, histtype='bar')#plt.legend(['Learned Prior', 'Proposal'])

plt.hist(temp2.cpu().numpy(), bins=[-5, -4, -3, -2, -1, 0], linewidth=1.2, histtype='step', stacked=True, fill=False)

plt.xlabel('Unscaled Selection (s)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof_hist2.png')
plt.close()

sns.kdeplot(df2, label=thecolumns)
#plt.legend(['Learned Prior', 'Proposal'])
#plt.legend()
plt.xlabel('Unscaled Selection (s)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof_log.png')
plt.close()

the_device='cuda:0'

box_uniform_prior = utils.BoxUniform(low=-0.015 * torch.ones(1, device=the_device), high=-1e-8*torch.ones(1,device=the_device),device=the_device)
accept_reject_fn = get_density_thresholder(obs20, quantile=1e-6)
proposal = RestrictedPrior(box_uniform_prior, accept_reject_fn, obs20, sample_with="sir", device=the_device)

dfe = proposal.sample((100000,))
temp3 = -1*torch.cat((sampleobs20, dfe),dim=1)
df3 = pd.DataFrame(temp3.cpu().numpy(), columns=['Restricted prior', 'Training Round 20'])
sns.kdeplot(df3, label=['Restricted Round 20', 'Training Round 20'])

plt.savefig('para_dfe.png')
