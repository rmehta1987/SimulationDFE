import torch
import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt


def get_sim_datafrom_hdf5(path_to_sim_file: str):
    """_summary_

    Args:
        path_to_sim_file (str): _description_
    """    
    #TODO probably will be better to use https://github.com/quantopian/warp_prism for faster look-up tables
    global loaded_file 
    global loaded_file_keys
    global loaded_tree
    import h5py
    loaded_file = h5py.File(path_to_sim_file, 'r')
    loaded_file_keys = list(loaded_file.keys())
    loaded_tree = KDTree(np.asarray(loaded_file_keys)[:,None]) # needs to have a column dimension


def get_sim_data(prior: float) -> torch.float32:

    _, idx = loaded_tree.query(prior.cpu().numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
    data = loaded_file[loaded_file_keys[idx[0]]]

    return torch.tensor(data).type(torch.float32)

def change_out_of_distance_proposals(prior: float):

    new_priors = [] 
    for j, a_prior in enumerate(prior):
        _, idx = loaded_tree.query(a_prior, k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
        cached_keys = float(loaded_file_keys[idx[0]])
        the_idx = np.abs(cached_keys - a_prior) > 1e-4
        if not the_idx:
            new_priors.append(a_prior)
        
    return new_priors


get_sim_datafrom_hdf5('sfs_lof_hdf5_data.h5')
load_emperical = np.load('emperical_lof_sfs_nfe.npy')
total_sites = np.sum(load_emperical)
num_samples = 12000
# Using α = 0.215, β = 562.1 from https://academic.oup.com/genetics/article/206/1/345/6064197
m = torch.distributions.Gamma(torch.tensor([0.215]), torch.tensor([562.1]))
sel_coef = -1*m.sample([num_samples])
correct_sel_coef = change_out_of_distance_proposals(sel_coef.numpy())

correct_sel_coef = torch.tensor(np.asarray(correct_sel_coef),dtype=torch.float32)






batches = torch.split(correct_sel_coef, int(num_samples/100))

for j, batch in enumerate(batches):
    data = torch.zeros_like(torch.tensor(load_emperical).type(torch.float32))
    for k, a_sel_coef in enumerate(sel_coef):
        data += get_sim_data(a_sel_coef)

    data = data / (k*1.0)
    torch.save(data, f'gamma_dfe+{k}.pkl')
        

