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
from sbi.utils import get_density_thresholder, RestrictedPrior, get_classifier_thresholder
from sbi.utils.user_input_checks_utils import CustomPriorWrapper
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
from sbi.samplers.vi.vi_pyro_flows import build_flow
from sbi.samplers.vi.vi_pyro_flows import get_flow_builder
from sbi.utils import mcmc_transform
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import MemReporter
from contextlib import redirect_stdout
from sbi.simulators.simutils import simulate_in_batches
from geomloss import SamplesLoss

from sbi.utils import RestrictionEstimator
import moments
# Flag Parser 
from absl import app 
from absl import flags

FLAGS = flags.FLAGS
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523

# Integer Flags
flags.DEFINE_integer('sample_size', 85, 'Diploid Population Sample Size where N is the number of diploids') # should be 55855
flags.DEFINE_integer('num_hidden',256, "Number of hidden layers in normalizing flow architecture")
flags.DEFINE_integer('num_sim', 300, 'How many simulations to run')
flags.DEFINE_integer('rounds', 16, 'How many round of simulations to run, (total simulations = num_sim*rounds')
flags.DEFINE_integer('seed', 10, 'A seed to set for reproducability')
flags.DEFINE_integer('number_of_transforms', 6, "How many normalizing flow blocks to use")
flags.DEFINE_integer('num_workers', 2, "How many workers to use for parallel simulations, be careful, can cause crashing")
flags.DEFINE_integer('num_blocks', 16, "How many blocks for sparse matrix")
flags.DEFINE_integer('num_bins', 8, "How many bins to use for rational spline normalizing flows")

flags.DEFINE_float('dropout_probability', 0.1, "Dropout probability")
flags.DEFINE_float('tail_bound', 8.0, "Bound for each spline only for spline flows")

# String Flags
flags.DEFINE_string('the_device', 'cuda', 'Whether to use CUDA or CPU')
flags.DEFINE_string('default_network_type', "nsf", "Type of normalizing flows architecture") # TODO change default network to integer discrete flows
flags.DEFINE_string('posterior_type', "VI", "Type of posterior to create (Direct, variational, MCMC, see SBI documentation for other flags") # TODO change default network to integer discrete flows
#Dataset path
flags.DEFINE_string('dataset_path', "/home/rahul/PopGen/SimulationSFS/sfs_missense_data.npy", "Where the dataset is located")


# Boolean Flags
flags.DEFINE_boolean("prior_returns_numpy", True, "If the prior of the simulator needs to be in numpy format")

plt.ioff() # so it doesn't show plots during debugging
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
    post._q_type = None
    post._q_arg = None

    return post

def get_state_classifier(post):
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
    post._q_type = None
    post._q_arg = None

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
    global num_blocks
    global num_bins
    global tail_bound
    global dropout_probability


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
    num_blocks = FLAGS.num_blocks
    num_bins = FLAGS.num_bins
    tail_bound = FLAGS.tail_bound
    dropout_probability = FLAGS.dropout_probability




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


def ImportanceSamplingEstimator(num_particles, sample, target, threshold, proposal2=None):
    """_summary_

    Args:
        num_particles (_type_): _description_
        sample (_type_): _description_
        target (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_
    """    
    cdftest = target.q.transforms
    low_samples = sample - .0025
    high_samples = sample + .0025 
    if proposal2 is not None:
        with torch.no_grad():
            for transform in cdftest[::-1]:
                value = transform.inv(high_samples)
            if target.q._validate_args:
                target.q.base_dist._validate_sample(value)
            value = target.q.base_dist.base_dist.cdf(value)
            #value = target.q._monotonize_cdf(value)
        with torch.no_grad():
            for transform in cdftest[::-1]:
                value2 = transform.inv(low_samples)
            if target.q._validate_args:
                target.q.base_dist._validate_sample(value2)
            value2 = target.q.base_dist.base_dist.cdf(value2)
            #value2 = target.q.base_dist._monotonize_cdf(value2)
        
        return value - value2
    else:
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

    data = np.zeros((sample_size*2-1))
    theprior = prior # last dim is misidentification
    #theprior=prior
    #mis_id = prior[-1].cpu().numpy()
    mis_id=0
    for a_prior in theprior:
        _, idx = loaded_tree.query(a_prior.cpu().numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
        fs = loaded_file[loaded_file_keys[idx[0]]][:]*15583.43726545000 #1164.3148344084038 # lof scaling parameter
        fs = (1 - mis_id)*fs + mis_id * fs[::-1]
        data += fs 
    data /= theprior.shape[0]
    return torch.log(torch.nn.functional.relu(torch.tensor(data, device=the_device))+1).type(torch.float32)

def generate_sim_data_only_one(prior: float) -> torch.float32:

    theprior=prior

    _, idx = loaded_tree.query(theprior.cpu().numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
    fs = loaded_file[loaded_file_keys[idx[0]]][:]*1164.3148344084038 # lof scaling parameter
  
    return torch.log(torch.nn.functional.relu(torch.tensor(fs, device=the_device)).type(torch.float32))

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
    #s_prior, weights = prior[:6], prior[6:]
    #s_prior, weights = prior[:5], prior[5:]
    #_prior, p_misid, weights = prior[:7], prior[7], prior[7:]
    s_prior, weights = prior[:20], prior[20:]
    fs_aggregate = None
    gammas = s_prior.cpu().numpy().squeeze()
    weights = weights.cpu().numpy().squeeze()
    #p_misid = p_misid.cpu().numpy()
    for j, (gamma, weight) in enumerate(zip(gammas, weights)):
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
        if j == 0:
            fs_aggregate = fs.project([projected_sample_size]).compressed()*theta_mis * weight
            #fs_aggregate = (1 - p_misid)*fs_aggregate + p_misid * fs_aggregate[::-1]
        else:
            fs2 = fs.project([projected_sample_size]).compressed()*theta_mis * weight
            #fs2 = (1 - p_misid)*fs2 + p_misid * fs2[::-1]
            fs_aggregate += fs2
            del fs2
            

    fs_aggregate = torch.poisson(torch.tensor(fs_aggregate)).type(torch.float32) 
    return fs_aggregate

def generate_moments_sim_data2(prior: float) -> torch.float32:
    
    global sample_size
    opt_params = [2.21531687, 5.29769918, 0.55450117, 0.04088086]
    theta_mis = 15583.437265450002
    theta_lof = 1164.3148344084038
    rerun = True
    ns_sim = 100
    h=0.5
    projected_sample_size = sample_size*2
    #s_prior, weights = prior[:6], prior[6:]
    #s_prior, weights = prior[:5], prior[5:]
    #s_prior, p_misid, weights = prior[:7], prior[7], prior[7:]
    fs_aggregate = None
    prior2 = prior[:-1]
    mis_id = prior[-1].cpu().numpy().squeeze()
    gammas = -1*10**(prior2.cpu().numpy().squeeze())
    nu_func = lambda t: [opt_params[0] * np.exp(
                np.log(opt_params[1] / opt_params[0]) * t / opt_params[3])]
    for j, gamma in enumerate(gammas):
        while rerun:
            ns_sim = 2 * ns_sim
            fs = moments.LinearSystem_1D.steady_state_1D(ns_sim, gamma=gamma, h=h)
            fs = moments.Spectrum(fs)
            fs.integrate([opt_params[0]], opt_params[2], gamma=gamma, h=h)
            
            fs.integrate(nu_func, opt_params[3], gamma=gamma, h=h)
            if abs(np.max(fs)) > 10 or np.any(np.isnan(fs)):
                # large gamma-values can require large sample sizes for stability
                rerun = True
                del fs
            else:
                rerun = False
        if j == 0:
            fs_aggregate = fs.project([projected_sample_size]).compressed()*theta_lof
            fs_aggregate = (1 - mis_id) * fs_aggregate + mis_id * fs_aggregate[::-1]
        else:
            fs2 = fs.project([projected_sample_size]).compressed()*theta_lof
            fs2 = (1 - mis_id) * fs2 + mis_id * fs2[::-1]
            fs_aggregate += fs2
        rerun = True
        ns_sim = 100
            
    fs_aggregate /= gammas.shape[0]
    fs_aggregate = torch.nn.functional.relu(torch.tensor(fs_aggregate)).type(torch.float32) 
    return fs_aggregate


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

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(X.device))[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def calibration_kernel_2(predicted, true_x, target, lossfunc):
    #lambda z: sinkhorn((z/z.sum(dim=1).view(z.shape[0],1)).unsqueeze(1), norm_true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1))
    #predicted = predicted.exp()
    
    norm_predicted = (predicted/predicted.sum(dim=1).view(predicted.shape[0],1)).unsqueeze(1)
    loss = lossfunc(norm_predicted, target.repeat(predicted.shape[0],1).to(the_device).unsqueeze(1))
    
   
    return loss

def restricted_simulations(proposal, true_x, simulator, sample_size=300, oversampling_factor=None):
    """restrict simulatins to be in a squared distance between true and simulated

    Args:
        theta (_type_): proposed selection coefficients
        predicted (_type_): simulated sfs
        true_x (_type_): emperical sfs
    """    

     #calculate l2 distance between predicted and true for the first 10 bins
    rerun = True
    truex10 = true_x[0,:10].unsqueeze(0)
    new_predicted = []
    new_theta = []
    rerun_counter = 0
    fail = False
    while rerun:
        
        if not oversampling_factor:
            theta = proposal.sample((30000,))
        else:
            theta = proposal.sample((10000,),oversampling_factor=oversampling_factor)
        #new_predicted = []
        predicted = simulate_in_batches(simulator, theta.to(the_device), num_workers=1, show_progress_bars=True)
        predicted10 = predicted[:,:10].exp()
        
        rmse = torch.nn.functional.mse_loss(predicted10.unsqueeze(1), truex10.repeat(predicted10.shape[0],1).to(the_device).unsqueeze(1),reduction='none')
        rmse = torch.sqrt(torch.mean(rmse,dim=2))
        trun_idx = torch.lt(rmse, 50.0*torch.ones_like(rmse)).squeeze(1)
        temp_theta = theta[trun_idx]
        temp_predicted = predicted[trun_idx]
        new_predicted.append(temp_predicted)
        new_theta.append(temp_theta)
        x = torch.cat(new_predicted, dim=0)
        if x.shape[0] >= sample_size:
            rerun = False
        else:
            rerun_counter += 1
            if rerun_counter > 200:
                fail=True
                print("Can't find good samples, rebuilding posterior")
                break
        print("Rerunning current, current shape of accepted simulations: {}".format(x.shape[0]))
    
    if not fail:
        good_theta = torch.cat(new_theta, dim=0)
        sns.kdeplot(good_theta.reshape(-1).cpu().numpy())
        plt.savefig('good_theta_check_{}.png'.format(torch.randint(1,1000,size=1)))
        return fail, good_theta, x
    else:
        return fail, None, None

def restricted_simulations2(proposal, true_x, simulator, sample_size=300, oversampling_factor=None):
    """restrict simulatins to be in a squared distance between true and simulated

    Args:
        theta (_type_): proposed selection coefficients
        predicted (_type_): simulated sfs
        true_x (_type_): emperical sfs
    """    

     #calculate l2 distance between predicted and true for the first 10 bins
    rerun = True
    truex10 = true_x[:,:30].unsqueeze(0)
    new_predicted = []
    new_theta = []
    bad_theta = []
    bad_predicted = []
    rerun_counter = 0
    fail = False
    while rerun:
        
        if not oversampling_factor:
            theta = proposal.sample((10000,))
        else:
            theta = proposal.sample((10000,),max_sampling_batch_size=5000, oversampling_factor=oversampling_factor)
        #new_predicted = []
        predicted = simulate_in_batches(simulator, theta.to(the_device), num_workers=1, show_progress_bars=True)
        predicted10 = predicted[:,:30].exp().unsqueeze(1)
        #predicted10 = predicted.exp().unsqueeze(1)
        
        rmse = torch.nn.functional.mse_loss(predicted10, truex10.repeat(predicted10.shape[0],1,1).to(the_device),reduction='none')
        rmse = torch.sqrt(torch.mean(rmse,dim=2))
        trun_idx = torch.lt(rmse, 80.0*torch.ones_like(rmse)).squeeze(1)
        temp_theta = theta[trun_idx]
        temp_predicted = predicted[trun_idx]
        # append the bad simulations
        bad_theta.append(theta[~trun_idx])
        bad_predicted.append(predicted[~trun_idx])
        # append the valid simulations
        new_predicted.append(temp_predicted)
        new_theta.append(temp_theta)

        # stop when we have enough good simulations
        x = torch.cat(new_predicted, dim=0)
        if x.shape[0] >= sample_size:
            rerun = False
            print("Finished collecting good samples, shape of accepted simulations: {}".format(x.shape[0]))
        if rerun:
            print("Rerunning to collect more samples, current shape of accepted simulations: {}".format(x.shape[0]))
        

    invalid_predicted = torch.cat(bad_predicted, dim=0)
    nan_predicted = torch.as_tensor([float("nan")],device=the_device)*torch.ones_like(invalid_predicted)
    good_predicted = torch.cat(new_predicted, dim=0)

    invalid_theta = torch.cat(bad_theta,dim=0)
    good_theta = torch.cat(new_theta, dim=0)
    if not oversampling_factor:
        final_theta = torch.cat((good_theta, invalid_theta[:150, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:150, :]),dim=0)
    elif invalid_theta.shape[0] > 1000:
        final_theta = torch.cat((good_theta, invalid_theta[:300, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:300, :]),dim=0)
    else:
        final_theta = torch.cat((good_theta, invalid_theta),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted),dim=0)
    return final_theta, final_x



def restricted_simulations3(proposal, true_x, simulator, sample_size=300, oversampling_factor=None):
    """restrict simulatins to be in a squared distance between true and simulated

    Args:
        theta (_type_): proposed selection coefficients
        predicted (_type_): simulated sfs
        true_x (_type_): emperical sfs
    """    

     #calculate l2 distance between predicted and true for the first 10 bins
    rerun = True
    truex10 = true_x[0,:30].unsqueeze(0)
    new_predicted = []
    new_theta = []
    bad_theta = []
    bad_predicted = []
    rerun_counter = 0
    fail = False
    while rerun:
        
        if not oversampling_factor:
            theta = proposal.sample((10000,), max_sampling_batch_size=5000)
        else:
            theta = proposal.sample((10000,),max_sampling_batch_size=5000, oversampling_factor=oversampling_factor)
        #new_predicted = []
        predicted = simulate_in_batches(simulator, theta.to(the_device), num_workers=1, show_progress_bars=True)
        predicted10 = predicted[:,:30].exp()
        
        rmse = torch.nn.functional.mse_loss(predicted10.unsqueeze(1), truex10.repeat(predicted10.shape[0],1).to(the_device).unsqueeze(1),reduction='none')
        rmse = torch.sqrt(torch.mean(rmse,dim=2))
        trun_idx = torch.lt(rmse, 50.0*torch.ones_like(rmse)).squeeze(1)
        temp_theta = theta[trun_idx]
        temp_predicted = predicted[trun_idx]
        # append the valid simulations
        new_predicted.append(temp_predicted)
        new_theta.append(temp_theta)

        # stop when we have enough good simulations
        x = torch.cat(new_predicted, dim=0)
        if x.shape[0] >= sample_size:
            rerun = False
            print("Finished collecting good samples, shape of accepted simulations: {}".format(x.shape[0]))
        if rerun:
            print("Rerunning to collect more samples, current shape of accepted simulations: {}".format(x.shape[0]))
        

    final_theta = torch.cat(new_theta, dim=0)[:sample_size,:]
    final_x= x[:sample_size,:]

    return final_theta, final_x

def main(argv):

    create_global_variables()
    set_reproducable_seed(FLAGS.seed)
    get_sim_datafrom_hdf5('moments_msl_sfs_lof_hdf5_data.h5')
    # Set uniform to [-5, -1e-4] selection < -5 is usually causes mutations to be at very low frequencies
    high_param = 4.0 * torch.ones(1, device=the_device)
    low_param = -6.0*torch.ones(1, device=the_device)

    simple_prior = torch.distributions.Uniform(low=low_param, high=high_param)

    ind_prior2 = MultipleIndependent(
    [
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        # torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=torch.zeros(1, device=the_device), high=1*torch.ones(1,device=the_device)),
    ],
    validate_args=False,)
    # Set up prior and simulator for SBI
    #prior, num_parameters, prior_returns_numpy = process_prior(box_uniform_prior)
    prior, num_parameters, prior_returns_numpy = process_prior(ind_prior2)
    #simulator = process_simulator(generate_moments_sim_data2, prior, prior_returns_numpy)
    simulator = process_simulator(generate_sim_data, prior, prior_returns_numpy)

    # Restricted Estimator:

    restriction_estimator = RestrictionEstimator(prior=ind_prior2, decision_criterion="nan", device=the_device)



    

    #simulator = generate_moments_sim_data
    # First learn posterior
    print("Setting up posteriors")
    density_estimator_function = posterior_nn(model="nsf", hidden_features=512, num_transforms=number_of_transforms, num_bins=num_bins, tail_bound=tail_bound, 
                                              dropout_probability=dropout_probability)

    true_x = load_true_data('emperical_missense_sfs_msl.npy', 0).unsqueeze(0)

    proposal = prior
    '''
    
    #true_x = load_true_data('emperical_missense_sfs_msl.npy', 0).unsqueeze(0)
    
    log_true_x = torch.log(true_x + 1)
    log_norm_true_x = log_true_x/log_true_x.sum(dim=1)
    norm_true_x = true_x/true_x.sum()
    print("True data shape (should be the same as the sample size): {} {}".format(true_x.shape[0], sample_size*2-1))
    path = "Experiments/saved_posteriors_msl_lof_scf_sinkhorn_12_pmsid_and_optimizer_{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    # Train posterior
    print("Starting to Train")
    
    #mmdloss = MMDLoss().to(the_device)
    #wassloss = SamplesLoss("energy", p=2, blur=0.05, scaling=0.8)True
    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.2, scaling=0.99)
    estimator_round = 10
    load_classifier=False
    # First created restricted estimator
    if load_classifier:
        restriction_estimator = torch.load('msl_restriction_classifier.pkl')
        proposal = restriction_estimator.restrict_prior(allowed_false_negatives=0.0)
        # for i in range(0,1):
        #     if i == 0:
        #         #theta = proposal.sample((2000,))
        #         theta, x = restricted_simulations2(proposal, true_x, simulator, sample_size=150)
        #     else:
        #         theta, x = restricted_simulations2(proposal, true_x, simulator, oversampling_factor=1024)
        #     restriction_estimator.append_simulations(theta, x)
        #     restriction_estimator.train(loss_importance_weights=True)
        #     proposal = restriction_estimator.restrict_prior(allowed_false_negatives=0.0)
    else:
        for i in range(0,estimator_round):
            if i == 0:
                #theta = proposal.sample((2000,))
                theta, x = restricted_simulations2(proposal, true_x, simulator, sample_size=150)
            else:
                theta, x = restricted_simulations2(proposal, true_x, simulator, oversampling_factor=1024)
            restriction_estimator.append_simulations(theta, x)
            if (i < estimator_round - 1):
                # training not needed in last round because classifier will not be used anymore.
                restriction_estimator.train(loss_importance_weights=True)
            proposal = restriction_estimator.restrict_prior(allowed_false_negatives=0.0)
            restriction_estimator._build_nn=None
            torch.save(restriction_estimator, 'msl_restriction_classifier_missense_6_coef.pkl')
    


    print("\nFinished creating restriction estimator\n")

    theta = proposal.sample((10000,), max_sampling_batch_size=10000)
    thetamin = theta.reshape(-1).min()
    thetamax = theta.reshape(-1).max()
    prior = CustomPriorWrapper(
            custom_prior=prior,
            event_shape=torch.Size([theta.numel()]),
            **custom_prior_wrapper_kwargs,
        )
    new_proposal = CustomPriorWrapper(proposal, event_shape=torch.Size([theta.shape[1]]), return_type=torch.float32, lower_bound=low_param, 
                                      upper_bound=high_param, arg_constraints=ind_prior2.custom_arg_constraints)
    #new_proposal = CustomPriorWrapper(proposal, event_shape=torch.Size([theta.shape[1]]), return_type=torch.float32, arg_constraints=ind_prior2.custom_arg_constraints)
    new_proposal, num_parameters, prior_returns_numpy = process_prior(new_proposal)
    
    theta = proposal.sample((300,), max_sampling_batch_size=5000)
    sns.kdeplot(theta.reshape(-1).cpu().numpy())
    plt.savefig('theta_round_inital.png')
    accept_reject_fn_classifier = get_classifier_thresholder(restriction_estimator._classifier, restriction_estimator._first_round_validation_theta, 
                                                             restriction_estimator._first_round_validation_label, 
                                                             allowed_false_negatives=0.0, reweigh_factor=None, device=the_device)
    '''
    infer_posterior = SNPE(proposal, show_progress_bars=True, device=the_device, density_estimator=density_estimator_function)
    for i in range(0,rounds+1):
        '''
        if i == 0:
            #theta = proposal.sample((300,), max_sampling_batch_size=5000)
            theta, x = restricted_simulations3(proposal, true_x, simulator, sample_size=300)
            sns.kdeplot(theta.reshape(-1).cpu().numpy())
            plt.savefig('theta_round_{}.png'.format(i))
            #fail, theta, x = restricted_simulations(proposal, true_x, simulator, sample_size=200)
        else:
            #theta = proposal.sample((300,), max_sampling_batch_size=5000, oversampling_factor=1024)
            theta, x = restricted_simulations3(proposal, true_x, simulator, sample_size=300)
            sns.kdeplot(theta.reshape(-1).cpu().numpy())
            plt.savefig('theta_round_{}.png'.format(i))
            #fail, theta, x = restricted_simulations(proposal, true_x, simulator, oversampling_factor=1024)
        
        #x = simulate_in_batches(simulator, theta.to(the_device), num_workers=1, show_progress_bars=True)

        '''
        print("\n******************Building density estimator for round {}*************************\n".format(i))

        if i == 0:
            theta = proposal.sample((300,))
            x = simulate_in_batches(simulator, theta.to(the_device), num_workers=1, show_progress_bars=True)
            #calibration_kernel = lambda z: torch.log(torch.sum(torch.nn.functional.mse_loss(z, true_x.repeat(z.shape[0],1).to(the_device),reduction='none'),dim=1))
            #calibration_kernel = lambda z: mmdloss(z.unsqueeze(1), true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1))
            #with torch.no_grad():
                #calibration_kernel = lambda z: sinkhorn((z/z.sum(dim=1).view(z.shape[0],1)).unsqueeze(1), norm_true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1)) + torch.nn.functional.poisson_nll_loss(z.unsqueeze(1),true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1))
                #calibration_kernel = lambda z: sinkhorn((z/z.sum(dim=1).view(z.shape[0],1)).unsqueeze(1), true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1))
                #calibration_kernel = lambda z: calibration_kernel_2(z, true_x, norm_true_x, sinkhorn)
            infer_posterior.append_simulations(theta, x.exp(), data_device='cpu', proposal=proposal ).train(learning_rate=5e-4, 
                                                                                   training_batch_size=30, use_combined_loss=True, calibration_kernel=None, 
                                                                                   show_train_summary=False,
                                                                                   force_first_round_loss=True)
        else:
            theta = proposal.sample((300,))
            x = simulate_in_batches(simulator, theta.to(the_device), num_workers=1, show_progress_bars=True)
            #with torch.no_grad():
                #calibration_kernel = lambda z: wassloss(z.unsqueeze(1), true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1))
                #calibration_kernel = lambda z: sinkhorn((z/z.sum(dim=1).view(z.shape[0],1)).unsqueeze(1), norm_true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1)) + torch.nn.functional.poisson_nll_loss(z.unsqueeze(1),true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1))
                #calibration_kernel = lambda z: sinkhorn((z/z.sum(dim=1).view(z.shape[0],1)).unsqueeze(1), norm_true_x.repeat(z.shape[0],1).to(the_device).unsqueeze(1))
                #calibration_kernel = lambda z: calibration_kernel_2(z, true_x, norm_true_x, sinkhorn)
            infer_posterior.append_simulations(theta, x.exp(), proposal=posterior_build, data_device='cpu').train(num_atoms=10, force_first_round_loss=True, 
                                                                                                            learning_rate=5e-4, training_batch_size=30,
                                                                                                            use_combined_loss=True, 
                                                                                                            show_train_summary=False, calibration_kernel=None)
        
        
        print("\n ****************************************** Building Posterior for round {} ******************************************.\n".format(i))
        
        if i == 0:
            #posterior parameters
            base_dist = torch.distributions.Independent(
                    torch.distributions.Normal(
                        0.0*torch.ones(ind_prior2._event_shape, device=the_device),
                        1.0*torch.ones(ind_prior2._event_shape, device=the_device),
                    ),
                    1,
                )
       
            vi_parameters = get_flow_builder("scf", batch_norm=False, base_dist = base_dist, permute = True, num_transforms=6,
                                             hidden_dims= [32, 32], skip_connections=False, nonlinearity=nn.ReLU(), count_bins=8, order="linear", bound=8 )
     
            posterior = infer_posterior.build_posterior(sample_with = "vi", prior=prior, vi_method="fKL", vi_parameters={"q": vi_parameters})
            print("\n ****************************************** Training to emperical observation for round {} ******************************************.\n".format(i))
            posterior_build = posterior.set_default_x(true_x).train(n_particles=200, max_num_iters=250, quality_control=False, stick_the_landing=True, alpha=0.5, 
                                                                    unbiased=True, dreg=True, K=12, learning_rate=5e-4, weight_decay=1e-4)
        #if i > 1 and fail:
        #    posterior = infer_posterior.build_posterior(sample_with = "vi", vi_method="fKL", vi_parameters={"q": vi_parameters})
        #    posterior_build = posterior.set_default_x(log_true_x).train(n_particles=500, retrain_from_scratch=True, max_num_iters=250, quality_control=False, stick_the_landing=True, alpha=0.5, unbiased=True, dreg=True)
        else:
            posterior = infer_posterior.build_posterior(sample_with = "vi", prior=prior, vi_method="IW", vi_parameters={"q": posterior_build})
            print("\n ****************************************** Training to emperical observation for round {} ******************************************.\n".format(i))
            posterior_build = posterior.set_default_x(true_x).train(n_particles=200, max_num_iters=250, quality_control=False, K=12, stick_the_landing=True, alpha=0.5, unbiased=True, dreg=True)

        
        # This proposal is used for Varitaionl inference posteior
        
        psi_metric = posterior_build.evaluate2(quality_control_metric= "psis", N=200)
        #prop_metric = posterior_build.evaluate2(quality_control_metric= "prop", N=250)
        print(f"\n****************Psi Metric is {psi_metric} and ideally should be less than 0.5.\n")
        #print(f"Prop Metric is {prop_metric} and ideally should be between 1.0 and 0.5 where 1.0 is best")
        #psi_metric = 3.0
        if i == 1:
            if not (os.path.isdir(path)):
                try:
                    os.mkdir(path)
                except OSError:
                    print("Error in making directory")    
        if psi_metric < 2.0 and i >= 1:
            # Save posteriors just in case
            print("PSI metric was really good! Preparting to save posteriors")
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

        
        accept_reject_fn_density = get_density_thresholder(posterior_build, quantile=1e-5, num_samples_to_estimate_support=100000)
        
        proposal_density = RestrictedPrior(new_proposal, accept_reject_fn_density, posterior_build, sample_with="sir", device=the_device)
        #proposal_classifier = RestrictedPrior(prior, accept_reject_fn_classifier, posterior_build, sample_with="sir", device=the_device)
        #proposal_classifier2 = RestrictedPrior(prior, accept_reject_fn_classifier, posterior_build, sample_with="rejection2", device=the_device)
        # Save posters every some rounds
        
        if i % 5 == 0 and i > 0:
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
        if i == rounds:
            try:
                path1 =path+"/inference_round_{}.pkl".format(i)
                with open(path1, "wb") as handle:
                    pickle.dump(infer_posterior, handle)
            except Exception:
                pass
            

            
            print("\n ****************************************** Saved Posterior for round {} ******************************************.\n".format(i))

    # Save posteriors and proposals for later use

    print("Finished Training posterior and prior")
    #torch.cuda.cudart().cudaProfilerStop()


    ### Currently saving objects does not seem to work for all posteriors/samplers (SRI, restricted estimator)
    '''
    # save Last posterior and observed posterior
    if posterior_type=='VI':
        path1 = path+"/posterior_last_round.pkl"
        path2 = path+"/proposal_last_round.pkl"
        path3 = path+"/posterior_observed_last_round.pkl"
        path4 = path+"/accept_function_last_round.pkl"
        with open(path1, "wb") as handle:
            post = get_state(posterior)
            torch.save(post, handle)
        
        # Can just re-estimate Restricted prior using sampling instead of saving
        # because it cannot save a copy of the get_density_thresholder function
        
        try:
            with open(path2, "wb") as handle:
                torch.save(proposals[i], handle)
        except:
            if i == 1:
                print("Cannot save proposal distributions")
            os.remove(path2)
        
        with open(path3, "wb") as handle:
            post = get_state(posterior_build)
            torch.save(post, handle)
    else:
        # Save posteriors just in case
        path1 = path+"/posterior_last_round.pkl"
        path3 = path+"/posterior_observed_last_round.pkl"
        with open(path1, "wb") as handle:
            torch.save(posterior, handle)
        with open(path3, "wb") as handle:
            torch.save(posterior_build, handle)
    
    try:
        with open("inference_laslt_round.pkl", "wb") as handle:
            pickle.dump(infer_posterior, handle)
    except Exception:
        pass
    '''
if __name__ == '__main__':
    app.run(main)
