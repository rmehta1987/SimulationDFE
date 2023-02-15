
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


class SummaryNet(nn.Module):
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.linear1 = sl.SparseLinear(self.sample_size, int(self.sample_size / 400))
        self.linear2 = sl.SparseLinear(int(self.sample_size / 400), int(self.sample_size / 400))
        self.linear3 = sl.SparseLinear(int(self.sample_size / 400), int(self.sample_size / 600))
        self.linear4 = sl.SparseLinear(int(self.sample_size / 600), int(self.sample_size / 800))
        self.linear5 = sl.SparseLinear(int(self.sample_size / 800), int(self.sample_size / 1200))

        self.model = nn.Sequential(self.linear1, nn.SiLU(), self.linear2, nn.SiLU(), self.linear3, nn.SiLU(), self.linear4, nn.SiLU(), self.linear5)

    def forward(self, x):
        
        x=self.model(x)
        return x
    

obs = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_missense_selection_1_2023-02-15_13-42/posterior_last_round.pkl')
obs1 = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_missense_selection_1_2023-02-15_13-42/posterior_observed_round_5.pkl')
obs0 = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_missense_selection_1_2023-02-15_13-42/posterior_observed_round_0.pkl')

sampleobs = obs.sample((100000,))
sampleobs1 = obs1.sample((100000,))
sampleobs0 = obs0.sample((100000,))

temp = torch.cat((sampleobs, sampleobs1, sampleobs0),dim=1)
temp = torch.log10(torch.abs(temp.squeeze()))
print(temp.shape)
df = pd.DataFrame(temp.cpu().numpy(), columns=['Learned Prior', 'Training Round 1', 'Initial Proposal'])
sns.kdeplot(df, label=['Learned Prior', 'Training Round 1', 'Initial Proposal'])
#plt.legend(['Learned Prior', 'Proposal'])
#plt.legend()
plt.xlabel('Unscaled Selection (s)')
plt.ylabel('Density')
plt.title('Density Selection Coefficients Sampled from Inferred Distributions of round')
plt.tight_layout()
plt.savefig('para_log.png')
plt.close()