from sortedcontainers import SortedDict
import numpy as np
import torch
from itertools import islice
from scipy.spatial import KDTree

dataset = np.load('sfs_missense_sim_data.npy', allow_pickle=True).item()

a = torch.distributions.Uniform(torch.tensor([1e-8]).double(),torch.tensor([.015]).double())
b = -1*a.sample([1000])
thekeys = list(dataset.keys())

tree = KDTree(np.asarray(thekeys)[:,None])

def closest(sorted_dict, key): # this is from stackoverflow but using kdtree is faster
    "Return closest key in `sorted_dict` to given `key`."
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


vals, idx = tree.query(b.numpy())

print(vals)
