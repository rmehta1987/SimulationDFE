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
#from pytorch_block_sparse import BlockSparseLinear
from sparselinear import activationsparsity as asy
from monarch_linear import MonarchLinear

from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import MemReporter
from contextlib import redirect_stdout
from sbi.simulators.simutils import simulate_in_batches

import moments
# Flag Parser 
from absl import app 
from absl import flags

FLAGS = flags.FLAGS
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523

# Integer Flags
flags.DEFINE_integer('sample_size', 85, 'Diploid Population Sample Size where N is the number of diploids') # should be 55855
flags.DEFINE_integer('num_hidden',256, "Number of hidden layers in normalizing flow architecture")
flags.DEFINE_integer('num_sim', 200, 'How many simulations to run')
flags.DEFINE_integer('rounds', 50, 'How many round of simulations to run, (total simulations = num_sim*rounds')
flags.DEFINE_integer('seed', 10, 'A seed to set for reproducability')
flags.DEFINE_integer('number_of_transforms', 3, "How many normalizing flow blocks to use")
flags.DEFINE_integer('num_workers', 2, "How many workers to use for parallel simulations, be careful, can cause crashing")
flags.DEFINE_integer('num_blocks', 16, "How many blocks for sparse matrix")

# String Flags
flags.DEFINE_string('the_device', 'cuda', 'Whether to use CUDA or CPU')
flags.DEFINE_string('default_network_type', "nsf", "Type of normalizing flows architecture") # TODO change default network to integer discrete flows
flags.DEFINE_string('posterior_type', "VI", "Type of posterior to create (Direct, variational, MCMC, see SBI documentation for other flags") # TODO change default network to integer discrete flows
#Dataset path
flags.DEFINE_string('dataset_path', "/home/rahul/PopGen/SimulationSFS/sfs_missense_data.npy", "Where the dataset is located")


# Boolean Flags
flags.DEFINE_boolean("prior_returns_numpy", True, "If the prior of the simulator needs to be in numpy format")


def get_state(post):
    """Need to use to save posterior if it is a variational posterior
        See: https://github.com/mackelab/sbi/issues/684
    Args:
        post (VIposterior): a VIPosterior object from SBI package

    Returns:
        post (VIposterior): a VIPosterior that can be pickled
    """    
    post._optimizer = None
    post.__deepcopy__ = None
    post._q_build_fn = None
    post._q.__deepcopy__ = None

    return post

def create_global_variables():
    """This creates a set of global variables to use in the entire program
    """

    # Setup devices and constants
    global the_device
    global sample_size
    global num_hidden
    global number_of_transforms
    global prior_returns_numpy
    global default_network_type
    global num_sim
    global rounds
    global posterior_type


    the_device = FLAGS.the_device
    if the_device and torch.cuda.is_available():
        print("The device is: {}".format(the_device))
    else:
        the_device = 'cpu'
        print("No CUDA device is: {}".format(the_device))

    sample_size = FLAGS.sample_size # Sample size of population, n, (56,885 is the sample size of non-Finnish European overall)
    num_hidden = FLAGS.num_hidden
    number_of_transforms = FLAGS.number_of_transforms
    prior_returns_numpy = FLAGS.prior_returns_numpy # Prior needs to be in numpy format for simulator
    default_network_type = FLAGS.default_network_type
    num_sim = FLAGS.num_sim
    rounds = FLAGS.rounds
    posterior_type = FLAGS.posterior_type




def set_reproducable_seed(the_seed: int):
    utils.sbiutils.seed_all_backends(the_seed) # set seed for reproducabilty

def get_sim_data(path_to_sim_file: str):
    """_summary_

    Args:
        path_to_sim_file (str): _description_
    """    
    #TODO probably will be better to use https://github.com/quantopian/warp_prism for faster look-up tables
    global loaded_file 
    global loaded_file_keys
    global loaded_tree
    loaded_file = np.load(f"{path_to_sim_file}", allow_pickle=True).item() # Loads sorted-dictionary
    loaded_file_keys = list(loaded_file.keys())
    loaded_tree = KDTree(np.asarray(loaded_file_keys)[:,None]) # needs to have a column dimension

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

def check_distance_from_cache_and_priors(proposed, path: str):

    if path:
        data = np.loadtxt(path, dtype=float)
        cached_keys = np.zeros_like(data)
        for j, adata in enumerate(data):
            _, idx = loaded_tree.query(adata, k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
            cached_keys[j] = float(loaded_file_keys[idx[0]])
        diff_cached_prior = np.subtract(cached_keys, data )
    else:
        count = 0
        for j, adata in enumerate(proposed):
            _, idx = loaded_tree.query(adata, k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
            cached_keys[j] = float(loaded_file_keys[idx[0]])
        diff_cached_prior = np.subtract(cached_keys, data )
    
    
    
    return np.square(np.subtract(data, cached_keys))


def ImportanceSamplingEstimator(num_particles, sample, target, threshold):
    """_summary_

    Args:
        num_particles (_type_): _description_
        sample (_type_): _description_
        target (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_
    """    

    
    sample_low = sample-threshold
    sample_high = sample+threshold
    proposal = torch.distributions.uniform.Uniform(sample_low, sample_high)
    prop_samps = proposal.sample((num_particles,))
    target_logprobs = target.log_prob(prop_samps)
    proposal_logprobs = proposal.log_prob(prop_samps)
    log_importance_weights = target_logprobs - proposal_logprobs

    ret = torch.sum(torch.exp(log_importance_weights))/num_particles


    return ret 

    
def generate_sim_data(prior: float) -> torch.float32:

    _, idx = loaded_tree.query(prior.cpu().numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
    data = loaded_file[loaded_file_keys[idx[0]]]

    return torch.poisson(torch.tensor(data, device=the_device)).type(torch.float32)

def change_out_of_distance_proposals(prior: float):
     
     need_to_learn = []

     for j, a_prior in enumerate(prior):
        _, idx = loaded_tree.query(a_prior, k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
        cached_keys = float(loaded_file_keys[idx[0]])
        the_idx = np.abs(cached_keys - a_prior) > 1e-4
        if the_idx:
            prior[j] = cached_keys
            need_to_learn.append(a_prior)
            

     return need_to_learn, torch.tensor(prior, dtype=torch.float32).unsqueeze(-1)

def generate_moments_sim_data(prior: float) -> torch.float32:
    
    global sample_size
    opt_params = [2.21531687, 5.29769918, 0.55450117, 0.04088086]
    theta_mis = 15583.437265450002
    theta_lof = 1164.3148344084038
    rerun = True
    ns_sim = 100
    h=0.5
    projected_sample_size = sample_size*2
    gamma = prior.cpu().numpy()
    
    while rerun:
        ns_sim = 2 * ns_sim
        fs = moments.LinearSystem_1D.steady_state_1D(ns_sim, gamma=gamma, h=h)
        fs = moments.Spectrum(fs)
        fs.integrate([opt_params[0]], opt_params[2], gamma=gamma, h=h)
        nu_func = lambda t: [opt_params[0] * np.exp(
            np.log(opt_params[1] / opt_params[0]) * t / opt_params[3])]
        fs.integrate(nu_func, opt_params[3], gamma=gamma, h=h)
        if abs(np.max(fs)) > 10 or np.any(np.isnan(fs)):
            # large gamma-values can require large sample sizes for stability
            rerun = True
        else:
            rerun = False


    fs = torch.tensor(fs).type(torch.float32)
    return fs

def generate_sim_data_from_hdf5(prior: float) -> torch.float32:

    _, idx = loaded_tree.query(prior.cpu().numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
    
    data = loaded_file[loaded_file_keys[idx[0]]][:]

    #return torch.cat([torch.poisson(torch.tensor(data, device=the_device)).type(torch.float32), torch.tensor([0.0],device=the_device)])
    return torch.poisson(torch.tensor(data, device=the_device)).type(torch.float32)


def true_sqldata_to_numpy(a_path: str, save_as: str):
    """_summary_

    Args:
        a_path (str): _description_
    """    
    ac_count = np.loadtxt(a_path, delimiter=",", dtype=object, skiprows=1)
    ac_count = ac_count[:,0].astype(float) # first column is the counts of alleles and converts to float

    thebins = np.arange(1,sample_size*2+1) 
    # Get the histogram
    sfs, bins = np.histogram(ac_count, bins=thebins)  
    assert sfs.shape[0] == sample_size*2-1, "True Sample Size must be the same dimensions as the Site Frequency Spectrum for the true sample size, SFS shape: {} and sample shape: {}".format(sfs.shape[0], sample_size*2-1)
    np.save('emperical_lof_sfs_nfe_ac_saved.pkl', sfs)

    print("Procssed and stored true data")

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
    assert sfs.shape[0] == sample_size*2-1, "Sample Size must be the same dimensions as the Site Frequency Spectrum, SFS shape: {} and sample shape (2*N-1): {}".format(sfs.shape[0], sample_size*2-1)

    return sfs 

@atexit.register
def finished_table_building():
    print("\n ****************************************** Completely finished experiment ****************************************** ")

def main(argv):

    create_global_variables()
    set_reproducable_seed(FLAGS.seed)
    

    box_uniform_prior = utils.BoxUniform(low=-0.2* torch.ones(10, device=the_device), high=-1e-5*torch.ones(10,device=the_device),device=the_device)
    # Set up prior and simulator for SBI
    prior, num_parameters, prior_returns_numpy = process_prior(box_uniform_prior)

    #simulator = process_simulator(generate_moments_sim_data, prior, prior_returns_numpy)

    simulator = generate_moments_sim_data
    # First learn posterior
    print("Setting up posteriors")
    density_estimator_function = posterior_nn(model="nsf", hidden_features=num_hidden, num_transforms=number_of_transforms)

    infer_posterior = SNPE(prior, show_progress_bars=True, device=the_device, density_estimator=density_estimator_function)

    #posterior parameters
    vi_parameters = {"q": "gaussian"}

    proposal = prior

    
    true_x = load_true_data('emperical_missense_sfs_msl.npy', 0).unsqueeze(0)
    print("True data shape (should be the same as the sample size): {} {}".format(true_x.shape[0], sample_size*2-1))
    path = "Experiments/saved_posteriors_msl_nsf{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    

    # Train posterior
    print("Starting to Train")
    num_ensembles = 10
    ensemble_pos= []
    for j in range(0,num_ensembles):
        print("Starting the {} ensamble".format(j))
        for i in range(0,rounds):

            theta = proposal.sample((num_sim,))
            x = simulate_in_batches(simulator, theta, num_workers=1, show_progress_bars=True)
            print("Inferring posterior for round {}\n".format(i))

            if i == 0:
                infer_posterior.append_simulations(theta, x,data_device='cpu' ).train(force_first_round_loss=True, learning_rate=5e-4, training_batch_size=10, use_combined_loss=True, show_train_summary=False)
            else:
                infer_posterior.append_simulations(theta, x, posterior_build, data_device='cpu' ).train(force_first_round_loss=True, learning_rate=5e-4, training_batch_size=10, use_combined_loss=True, show_train_summary=True)

            print("\n ****************************************** Building Posterior for round {} ******************************************.\n".format(i))

            posterior = infer_posterior.build_posterior(sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)

            print("Training to emperical observation")
            # This proposal is used for Varitaionl inference posteior
            posterior_build = posterior.set_default_x(true_x).train(n_particles=200, max_num_iters=500, quality_control=False)
            prop_metric = posterior_build.evaluate2(quality_control_metric= "prop", N=200)
            psi_metric = posterior_build.evaluate2(quality_control_metric= "psis", N=200)
            print(f"Psi Metric is {psi_metric} and ideally should be less than 0.5.  The Prop Metric is {prop_metric} and ideally should be greater than 0.5, where 1.0 is best")
            if (psi_metric < 0.0  and prop_metric < 0.5) or (psi_metric > 0.5):
                print("Retraining posterior because it is not proportial to the potential function")
                posterior_build = posterior.set_default_x(true_x).train(learning_rate=1e-4 * 0.1, retrain_from_scratch=True,reset_optimizer=True, quality_control=False, n_particles=100)
                psi_metric = posterior_build.evaluate2(quality_control_metric= "psis", N=200)
                prop_metric = posterior_build.evaluate2(quality_control_metric= "prop", N=200)
                print("Psi Metric is {} and ideally should be less than 0.5.  The Prop Metric is {} and ideally should be greater than 0.5, where 1.0 is best".format(psi_metric, prop_metric))
            
            accept_reject_fn = get_density_thresholder(posterior_build, quantile=1e-5, num_samples_to_estimate_support=1000)
            proposal = RestrictedPrior(prior, accept_reject_fn, posterior_build, sample_with="sir", device=the_device)

        ensemble_pos.append(posterior_build)

    # Train weights of ensamble
    learnable_weights =  torch.nn.Parameter(torch.Tensor(1, num_ensembles, 1, device=the_device), requires_grad=True).fill_(1. / num_ensembles)  
    nposteriors = NeuralPosteriorEnsemble(ensemble_pos, weights=learnable_weights)
    nposteriors.set_default_x(true_x)
    e_opt = torch.optim.Adam(learnable_weights.parameters(), lr=0.01)
    while (i <= iter):

        with torch.no_grad():
            samples = nposteriors.sample((1000,))
            log_prob, log_weights = nposteriors.log_prob(samples, individually=True)
        

        totallp = torch.logsumexp(log_weights.expand_as(log_prob) + log_prob, dim=0)

        
        






        # Save posters every some rounds
        '''
        if i == 0:
            if not (os.path.isdir(path)):
                try:
                    os.mkdir(path)
                except OSError:
                    print("Error in making directory")    
        if i % 5 == 0:
            print("Preparting to save posteriors")
            if posterior_type == "VI":
                # Save posteriors just in case
                path1 = path+"/posterior_round_{}.pkl".format(i)
                path3 = path+"/posterior_observed_round_{}.pkl".format(i)
                with open(path1, "wb") as handle:
                    temp = posterior
                    post = get_state(temp)
                    torch.save(post, handle)
                with open(path3, "wb") as handle:
                    temp2 = posterior_build
                    post = get_state(temp2)
                    torch.save(post, handle)
            else:
                # Save posteriors just in case
                path1 = path+"/posterior_round_{}.pkl".format(i)
                path3 = path+"/posterior_observed_round_{}.pkl".format(i)
                with open(path1, "wb") as handle:
                    torch.save(posterior, handle)
                with open(path3, "wb") as handle:
                    torch.save(posterior_build, handle)
        if i % 20 == 0 and i > 0:
            try:
                path1 =path+"/inference_round_{}.pkl".format(i)
                with open(path1, "wb") as handle:
                    pickle.dump(infer_posterior, handle)
            except Exception:
                pass
        '''
            

            
        print("\n ****************************************** Saved Posterior for round {} ******************************************.\n".format(i))


if __name__ == '__main__':
    app.run(main)
