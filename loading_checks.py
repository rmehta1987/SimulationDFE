
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


        self.model = nn.Sequential(self.linear4, self.bn4, asy.ActivationSparsity(),
                                   self.linear5, self.bn5, asy.ActivationSparsity(),
                                   self.linear6, self.bn6, asy.ActivationSparsity(),
                                   self.linear7, self.bn7, asy.ActivationSparsity(),
                                   self.linear8,)
                                   
                                   

    def forward(self, x):
        
        x=self.model(x)
        return x
the_device='cuda:0'

proposal = utils.BoxUniform(low=-0.015 * torch.ones(1, device=the_device), high=-1e-8*torch.ones(1,device=the_device),device=the_device)

saved_path='saved_posteriors_nfe_infer_lof_selection_1_gpu2023-02-23_22-24'
obs30 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_30.pkl')
obs25 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_25.pkl')
obs20 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_20.pkl')
obs15 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_15.pkl')
obs10 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_10.pkl') 
obs5 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_5.pkl')
obs0 = torch.load(f'Experiments/{saved_path}/posterior_observed_round_0.pkl')
sampleobs30 = obs30.sample((100000,))
sampleobs25 = obs25.sample((100000,))

sampleobs20 = obs20.sample((100000,))
sampleobs15 = obs15.sample((100000,))
sampleobs10 = obs10.sample((100000,))
sampleobs5 = obs5.sample((100000,))
sampleobs0 = obs0.sample((100000,))
obsp = proposal.sample((100000,))

bins = [0, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]

temp = -1*torch.cat((sampleobs30, sampleobs25, sampleobs20, sampleobs15, sampleobs10, sampleobs5, sampleobs0),dim=1)
#temp = -1*torch.cat((sampleobs20, sampleobs15, sampleobs10, sampleobs5, sampleobs0),dim=1)
#temp = -1*torch.cat((sampleobs10, sampleobs5, sampleobs0, obsp),dim=1)
temp2 = torch.log10(torch.abs(temp.squeeze()))
print(temp.shape)
thecolumns = ['Posterior Round {}'.format(i*5) for i in range(1, temp.shape[1])]
thecolumns.insert(0, 'Inital Proposal')
thecolumns.reverse()
print(thecolumns)
df = pd.DataFrame(temp.cpu().numpy(), columns=thecolumns)
df2 = pd.DataFrame(temp2.cpu().numpy(), columns=thecolumns)
sns.kdeplot(df, label=thecolumns)
#plt.legend(['Learned Prior', 'Proposal'])
#plt.legend()
plt.xlabel('Unscaled Selection (|s|)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_lof_30.png')
plt.close()

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


box_uniform_prior = utils.BoxUniform(low=-0.015 * torch.ones(1, device=the_device), high=-1e-8*torch.ones(1,device=the_device),device=the_device)
accept_reject_fn = get_density_thresholder(obs10, quantile=1e-6)
proposal = RestrictedPrior(box_uniform_prior, accept_reject_fn, obs10, sample_with="sir", device=the_device)

dfe = proposal.sample((100000,))
temp3 = -1*torch.cat((dfe, sampleobs30, obsp),dim=1)
df3 = pd.DataFrame(temp3.cpu().numpy(), columns=['Restricted prior', 'Training Round 30', 'Initial Propsal'])
temp4 = torch.log10(torch.abs(temp3.squeeze()))

sns.kdeplot(df3, label=['Restricted Round 30', 'Training Round 30', 'Initial Proposal'])
plt.xlabel('Unscaled Selection (|s})')
plt.ylabel('Density')
plt.savefig('para_dfe_30.png')
plt.close()
temp4 = torch.log10(torch.abs(temp3.squeeze()))
df4 = pd.DataFrame(temp4.cpu().numpy(), columns=['DFE Round 30', 'Training Round 30', 'Initial Propsal'])

sns.kdeplot(df4, label=['DFE Round 30', 'Training Round 30', 'Initial Proposal'])
plt.xlabel('Unscaled Selection (log(|s|)})')
plt.ylabel('Density')
plt.savefig('para_dfe_log_30.png')
