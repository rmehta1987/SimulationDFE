import yaml
import os
import sys
import argparse
import subprocess
import datetime

import torch


from vae_sim_sfs import VAE
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks
import h5py
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
from pyro.nn import AutoRegressiveNN, DenseNN
from torch.distributions import Distribution, Independent, biject_to, constraints

from pyro.distributions import transforms

def createFlow(prior):
    
    count_bins = 8
    dim=10
    param_dims = [count_bins, count_bins, (count_bins - 1), count_bins]
    device = 'cuda'
    base_dist=None
    num_transforms = 6
    bound=8 
    neural_net = AutoRegressiveNN(
        dim=10,
        hidden_dims=128,
        param_dims=param_dims,
        skip_connections=False,
        nonlinearity=torch.nn.ReLU(),
    ).to(device)
    permute=True
    batch_norm=False
    if base_dist is None:
        base_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(dim, device=device),
                torch.ones(dim, device=device),
            ),
            1,
        )
  

    flows = []
    for i in range(num_transforms):
        flows.append(
            transforms.SplineAutoregressive(dim, neural_net, count_bins, bound).with_cache()
        )
        if permute and i < num_transforms - 1:
            permutation = torch.randperm(dim, device=device)
            flows.append(transforms.Permute(permutation))
        if batch_norm and i < num_transforms - 1:
            bn = transforms.BatchNorm(dim).to(device)
            flows.append(bn)

    prior_transform = biject_to(prior.support)
    flows.append(prior_transform)
    dist = transforms.TransformedDistribution(base_dist, flows)
    return dist

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


def generate_sim_data(prior: float, scaling_param) -> torch.float32:


    
    for j, a_prior in enumerate(prior):

        _, idx = loaded_tree.query(prior.numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
        data = loaded_file[loaded_file_keys[idx[0]]][:]*scaling_param
        if j > 0:
            data += loaded_file[loaded_file_keys[idx[0]]][:]*scaling_param

    data /= prior.shape[0]
    return torch.nn.functional.relu(torch.tensor(data)).type(torch.float32) 

@torch.no_grad()
def test(model, prob_model, loader, device):
    model.eval()
    mask_bc = loader.dataset[:][1].to(device)
    generated_data = model([loader.dataset[:][0].to(device), mask_bc, None], mode=False).cpu()

    data = loader.dataset[:][0]
    plt.plot_together([data, generated_data], prob_model, title='', legend=['original', 'generated'],
                      path=f'{args.root}/marginal')

def sim_data(prior):
    sim_data = []
    for a_sample in prior:
        sim_data.append(generate_sim_data(a_sample))

    return torch.cat(sim_data)

def main(hparams):
    os.makedirs(hparams.root, exist_ok=True)

    if hparams.to_file:
        sys.stdout = open(f'{hparams.root}/stdout.txt', 'w')
        sys.stderr = open(f'{hparams.root}/stderr.txt', 'w')


    if hparams.latent_size is None:
        if hparams.latent_perc is not None:
            hparams.latent_size = max(1, int(len(prob_model.gathered) * (hparams.latent_perc / 100) + 0.5))
        else:
            hparams.latent_size = max(1, int(len(prob_model.gathered) * 0.75 + 0.5))

    if not hasattr(hparams, 'size_s') or hparams.size_s is None:
        hparams.size_s = hparams.latent_size

    if not hasattr(hparams, 'size_z') or hparams.size_z is None:
        hparams.size_z = hparams.latent_size

    if not hasattr(hparams, 'size_y') or hparams.size_y is None:
        hparams.size_y = hparams.hidden_size

   
    model = VAE(hparams)

    tb_logger = None
    if hparams.tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(f'{hparams.root}/tb_logs')

    timer = pl_callbacks.Timer()
    checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=hparams.root, filename='best',
                                                       monitor='validation/re', save_last=True)

    high_param = 5.0 * torch.ones(10, device=hparams.device)
    low_param = -6.0*torch.ones(10, device=hparams.device)
    prior = torch.distributions.Uniform(low=low_param, high=high_param)
    flows = createFlow(prior)
    for epochs in range(0, 20):
        
        thedata = sim_data(prior)
        loss, logs = model(thedata)
        loss.backward()





    seconds = timer.time_elapsed('train')
    print(f'Training finished in {int(seconds)}s ({datetime.timedelta(seconds=seconds)}).')

    


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    # Configuration
    parser = argparse.ArgumentParser('')

    # General
    parser.add_argument('-seed', type=int, default=None)
    parser.add_argument('-device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('-root', type=str, default='.', help='Output folder (default: \'%(default)s)\'')
    parser.add_argument('-to-file', action='store_true', help='Redirect output to \'stdout.txt\'')

    parser.add_argument('-model', type=str, required=True, choices=['vae', 'iwae', 'hivae', 'dreg'])

    # Tracking
    parser.add_argument('-tensorboard', action='store_true', help='Activates tensorboard logs.')


    # Training
    group = parser.add_argument_group('training')
    group.add_argument('-learning-rate', type=float, default=0.0001, help='Learning rate')
    group.add_argument('-decay', type=float, default=1., help='Learning rate\'s exponential decay rate.')  # 0.999999
    group.add_argument('-max-epochs', type=int, default=None, help='Number of epochs.')

    # VAE
    group = parser.add_argument_group('vae/iwae')
    group.add_argument('-latent-size', type=int, default=50)
    group.add_argument('-latent-perc', type=int, default=None)
    group.add_argument('-dropout', type=float, default=0.1, help='Dropout percentage on the input layer')
    group.add_argument('-hidden-size', type=int, default=256, help='Size of the hidden layers')

    args = parser.parse_args()
    main(args)

    sys.exit(0)