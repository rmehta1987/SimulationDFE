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
from functools import partial

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import torch.distributions as dists
from torch.nn.functional import softplus
from torch.distributions import constraints
from torch.distributions.utils import logits_to_probs


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

def create_a_dfe(path: str):

    create_gamma_dfe(path)

def init_weights(m, gain=1.):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.01)

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, dropout):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout), nn.BatchNorm1d(input_size),
            nn.Linear(input_size,  hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

        self.z_loc = nn.Linear(hidden_size, latent_size)
        self.z_log_scale = nn.Linear(hidden_size, latent_size)

        self.encoder.apply(partial(init_weights, gain=nn.init.calculate_gain('tanh')))
        self.z_loc.apply(init_weights)
        self.z_log_scale.apply(init_weights)

    def q_z(self, loc, logscale):
        scale = softplus(logscale)
        return dists.Normal(loc, scale)

    def forward(self, x):
        h = self.encoder(x)

        loc = self.z_loc(h)  # constraints.real
        log_scale = self.z_log_scale(h)  # constraints.real

        return loc, log_scale


class VAE(pl.LightningModule):
    def __init__(self, input_size, hparams, true_x):
        super().__init__()

        self.save_hyperparameters()
        self.prior_z_loc = torch.zeros(hparams.latent_size)
        self.prior_z_scale = torch.ones(hparams.latent_size)
        self.true_x = true_x
        hparams = self.hparams

        # encoder, decoder
        self.encoder = Encoder(input_size, hparams.latent_size, hparams.hidden_size, hparams.dropout)
        self.decoder_shared = nn.Sequential(
            nn.Linear(hparams.latent_size, hparams.hidden_size), nn.ReLU(),
            nn.Linear(hparams.hidden_size, hparams.hidden_size), nn.ReLU(),
            nn.Linear(hparams.hidden_size, hparams.hidden_size), nn.ReLU(),
        )

        # distribution parameters
        self.decoder_shared.apply(partial(init_weights, gain=nn.init.calculate_gain('relu')))

    @property
    def prior_z(self):
        return dists.Normal(self.prior_z_loc, self.prior_z_scale)

    def _run_step(self, x):

        # Generate Encoder
        z_params = self.encoder(x)

        # # Generate approximate posterior via Encoder
        z = self.encoder.q_z(*z_params).rsample([self.samples])

        x_hat = self.decoder_shared(z)
      
        # samples x batch_size x D
        log_px_z = self.log_likelihood(x_hat)

        log_pz = self.prior_z.log_prob(z).sum(dim=-1)  # samples x batch_size
        log_qz_x = self.encoder.q_z(*z_params).log_prob(z).sum(dim=-1)  # samples x batch_size
        kl_z = log_qz_x - log_pz

        return log_px_z, kl_z
    
    def _step(self, batch, batch_idx):
        x, _= batch
        log_px_z, kl_z = self._run_step(x)

        elbo = sum(log_px_z) - kl_z
        loss = -elbo.squeeze(dim=0).sum(dim=0)
        assert loss.size() == torch.Size([])

        logs = dict()
        logs['loss'] = loss / x.size(0)

        with torch.no_grad():
            log_prob = (self.log_likelihood_real(self.true_x)).sum(dim=0)
            logs['re'] = -log_prob.mean(dim=0)
            logs['kl'] = kl_z.squeeze(dim=0).mean(dim=0)
            logs.update({f'll_{i}': l_i.item() for i, l_i in enumerate(log_prob)})

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'training/{k}': v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'validation/{k}': v for k, v in logs.items()})
        return loss
    
    def _infer_step(self, x, mode):
        z_params = self.encoder(x)
        if mode:
            z = z_params[0]  # Mode of a Normal distribution
        else:
            z = self.encoder.q_z(*z_params).sample()

        x_hat = self.decoder_shared(z)

        return x_hat
    
    def forward(self, batch, mode=True):
        x, _ = batch
        return self._infer_step(x, mode=mode)[0]
    
    # Measures
    def log_likelihood(self, x, x_hat):
        log_prob = torch.nn.PoissonNLLLoss()
        output = log_prob(torch.log(x_hat), torch.log(x))

        return output
   
    def log_likelihood_real(self, x):
        x_params = self._infer_step(x, mode=True)
        return self.log_likelihood(x, x_params)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.parameters(), 'lr': self.hparams.learning_rate},
        ])

        if self.hparams.decay == 1.:
            return optimizer

        # We cannot set different schedulers if we want to avoid manual optimization
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.decay)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'  # Alternatively: "step"
            },
        }

vae = VAE()



x_encoded = vae.encoder(x)
mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)



    

