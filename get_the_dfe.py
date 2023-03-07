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
from sortedcontainers import SortedDict
from scipy.spatial import KDTree
from pytorch_block_sparse import BlockSparseLinear
from sparselinear import activationsparsity as asy
from monarch_linear import MonarchLinear

from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import MemReporter
from contextlib import redirect_stdout
from sbi.inference import posterior_estimator_based_potential, MCMCPosterior

def load_true_data(a_path: str, type: int) -> torch.float32:
    """Loads a true SFS, note that the sample size must be consistent with the passed parameters

    Args:
        path (str): Where the true-SFS is located, must be a numpy array
        type (int): is data stored in numpy pickle (0) or torch pickle (1)

    Returns:
        Returns the SFS of the true data-set
    """
    if type == 0:
        sfs = np.load(a_path)
        sfs = torch.tensor(sfs, device=the_device).type(torch.float32)
    else:
        sfs = torch.load(a_path)
        sfs.to(the_device)

    return sfs 

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
the_device='cuda:0'

prior = utils.BoxUniform(low=-0.990 * torch.ones(1, device=the_device), high=-1e-8*torch.ones(1,device=the_device),device=the_device)
post90 = torch.load('/home/rahul/PopGen/SimulationSFS/Experiments/saved_posteriors_nfe_infer_lof_selection_monarch_2023-03-02_16-13/posterior_round_65.pkl')

accept_reject_fn = get_density_thresholder(post90, quantile=1e-6)
proposal = RestrictedPrior(prior, accept_reject_fn, post90, sample_with="sir", device=the_device)


true_x = load_true_data('emperical_lof_sfs_nfe.npy', 0)

potential_fn, parameter_transform = posterior_estimator_based_potential(
    post90, prior, x_o = true_x)

dfe = MCMCPosterior(
    potential_fn, proposal=proposal, theta_transform=parameter_transform
)