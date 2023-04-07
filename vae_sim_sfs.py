import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import seaborn as sns
import datetime
import pandas as pd
import torch.nn.functional as F
import moments
import h5py
from scipy.spatial import KDTree
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523


def get_sim_datafrom_hdf5(path_to_sim_file: str):
    """_summary_

    Args:
        path_to_sim_file (str): _description_
    """    
    #TODO probably will be better to use https://github.com/quantopian/warp_prism for faster look-up tables
    global loaded_file 
    global loaded_file_keys
    global loaded_tree
    loaded_file = h5py.File(path_to_sim_file, 'r')
    loaded_file_keys = list(loaded_file.keys())
    loaded_tree = KDTree(np.asarray(loaded_file_keys)[:,None]) # needs to have a column dimension


def generate_sim_data(prior: float) -> torch.float32:

    if prior < -5.0:
        _, idx = loaded_tree.query(-5.0, k=(1,))
    else:
        _, idx = loaded_tree.query(prior.numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
    data = loaded_file[loaded_file_keys[idx[0]]][:]

    return data



def create_gamma_dfe(file_path, alpha=0.1596, beta=2332.3, size_of_data_set=10000, num_particles=1000, scaling_factor=15583.437265450002):

    get_sim_datafrom_hdf5(file_path)

    data = None
    #shape: 0.1596
    #scale: 2332.3
    gamma_dist = torch.distributions.gamma.Gamma(torch.tensor(alpha),torch.tensor(1/beta))
    dataset = [None]*size_of_data_set
    for i in range(1, size_of_data_set):
        samples = -1*gamma_dist.sample((num_particles,))
        for j, sample in enumerate(samples):
            if j == 0:
                data = generate_sim_data(sample)*scaling_factor
            else:
                data += generate_sim_data(sample)*scaling_factor
        data /= num_particles
        dataset[i]=torch.poisson(torch.tensor(data)).type(torch.float32)
    
    with open('moments_gamma_dfe_data.pkl', "wb") as handle:
        torch.save(dataset, handle )



create_gamma_dfe('moments_msl_sfs_lof_hdf5_data.h5')




    

