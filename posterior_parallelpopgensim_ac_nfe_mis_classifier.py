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
from sortedcontainers import SortedDict
from scipy.spatial import KDTree
#from pytorch_block_sparse import BlockSparseLinear
from sparselinear import activationsparsity as asy
from monarch_linear import MonarchLinear
from sbi.samplers.vi.vi_pyro_flows import build_flow
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import MemReporter
from contextlib import redirect_stdout
from sbi.simulators.simutils import simulate_in_batches
from geomloss import SamplesLoss
from sbi.samplers.vi.vi_pyro_flows import build_flow
from sbi.samplers.vi.vi_pyro_flows import get_flow_builder
from sbi.utils.user_input_checks_utils import CustomPriorWrapper
from copy import deepcopy
from tqdm import tqdm

from sbi.utils import RestrictionEstimator
# Flag Parser
from absl import app
from absl import flags

FLAGS = flags.FLAGS
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523

# Integer Flags
flags.DEFINE_integer('sample_size', 55855, 'Diploid Population Sample Size where N is the number of diploids') # should be 55855
flags.DEFINE_integer('num_hidden',256, "Number of hidden layers in normalizing flow architecture")
flags.DEFINE_integer('num_sim', 200, 'How many simulations to run')
flags.DEFINE_integer('rounds', 15, 'How many round of simulations to run, (total simulations = num_sim*rounds')
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

def aggregated_generate_sim_data(prior: float) -> torch.float32:

    data = np.zeros((sample_size*2-1))
    theprior = prior[:-1] # last dim is misidentification
    gammas = -1*10**(theprior.cpu().numpy().squeeze())

    mis_id=0
    for a_prior in gammas:
        _, idx = loaded_tree.query(a_prior, k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
        fs = loaded_file[loaded_file_keys[idx[0]]][:]
        fs = (1 - mis_id)*fs + mis_id * fs[::-1]
        fs = fs*.2077370846  # scale to missense theta
        data += fs
    data /= theprior.shape[0]
    return torch.log(torch.nn.functional.relu(torch.tensor(data)+1).type(torch.float32))

def generate_sim_data_only_one(prior: float) -> torch.float32:

    theprior=prior

    _, idx = loaded_tree.query(theprior.cpu().numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
    fs = loaded_file[loaded_file_keys[idx[0]]][:] # lof scaling parameter

    return torch.log(torch.nn.functional.relu(torch.tensor(fs))+1).type(torch.float32)


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
        sfs = torch.tensor(sfs).type(torch.float32)
    else:
        sfs = torch.load(a_path)
        sfs.to(the_device)
    assert sfs.shape[0] == sample_size*2-1, "Sample Size must be the same dimensions as the Site Frequency Spectrum, SFS shape: {} and sample shape (2*N-1): {}".format(sfs.shape[0], sample_size*2-1)

    return sfs


class SummaryNet(nn.Module):
    def __init__(self, sample_size, block_sizes, dropout_rate=0.0):
        super().__init__()
        self.sample_size = sample_size # For monarch this needs to be divisible by the block size
        self.block_size = block_sizes
        self.linear4 = MonarchLinear(self.sample_size, int(self.sample_size / 10), nblocks=self.block_size[0]) # 11171
        self.linear5 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 10) , nblocks=self.block_size[1]) # 11171
        self.linear6 = MonarchLinear(int(self.sample_size / 10), int(self.sample_size / 10), nblocks=self.block_size[2]) # 11171

        self.model = nn.Sequential(self.linear4, nn.Dropout(dropout_rate), nn.GELU(),
                                   self.linear5, nn.Dropout(dropout_rate), nn.GELU(),
                                   self.linear6)
    def forward(self, x):

        x=self.model(x)
        return x

@atexit.register
def finished_table_building():
    print("\n ****************************************** Completely finished experiment ****************************************** ")

def restricted_simulations(proposal, true_x, simulator, sample_size=300, oversampling_factor=None):
    """restrict simulatins to be in a squared distance between true and simulated

    Args:
        theta (_type_): proposed selection coefficients
        predicted (_type_): simulated sfs
        true_x (_type_): emperical sfs
    """

     #calculate l2 distance between predicted and true for the first 10 bins
    rerun = True
    truex10 = true_x[:,1:40].unsqueeze(0).to('cpu')
    new_predicted = []
    new_theta = []
    bad_theta = []
    bad_predicted = []
    rerun_counter = 0
    fail = False
    while rerun:

        if not oversampling_factor:
            theta = proposal.sample((1000,))
        else:
            theta = proposal.sample((1000,),max_sampling_batch_size=5000, oversampling_factor=oversampling_factor)
        #new_predicted = []
        predicted = simulate_in_batches(simulator, theta, num_workers=1, show_progress_bars=True).to('cpu')
        predicted10 = predicted[:,1:40].to('cpu').exp().unsqueeze(1)
        #predicted10 = predicted.exp().unsqueeze(1)
        #import pdb
        #pdb.set_trace()
        rmse = torch.nn.functional.mse_loss(predicted10, truex10.repeat(predicted10.shape[0],1,1),reduction='none').squeeze(1) # take out the 2nd column
        sorted_rmse = torch.argsort(rmse, dim=0)[0] # find the best rmse along the rows
        best_rows = torch.cat((sorted_rmse.unsqueeze(1), torch.arange(0,sorted_rmse.shape[0]).unsqueeze(1)),dim=1) # since we sorted along rows, we need to identify which columns correspond to the bins
        rmse_best = torch.sqrt(rmse[best_rows[:,0],best_rows[:,1]])
        #rmse = torch.sqrt(torch.mean(rmse,dim=2))
        trun_idx = torch.lt(rmse_best, 150.0*torch.ones_like(rmse_best))
        #goodtheta = theta.squeeze(1)best2[trun][:,0]
        best_idx = best_rows[trun_idx]
        best_idx_unique = best_idx[:,0].unique()
        temp_theta = theta[best_idx_unique]
        temp_predicted = predicted[best_idx_unique,:]
        # find the bad ones
        sorted_rmse = torch.argsort(rmse, dim=0)[-10:].reshape(-1).unique() # find the worst rmse along the rows, don't care about columns since they are all very bad rmse
        # append the bad simulations
        bad_theta.append(theta[sorted_rmse].to('cpu'))
        bad_predicted.append(predicted[sorted_rmse].to('cpu'))
        # append the valid simulations
        new_predicted.append(temp_predicted)
        new_theta.append(temp_theta)

        # stop when we have enough good simulations
        x = torch.cat(new_predicted, dim=0)
        if x.shape[0] >= sample_size:
            rerun = False
            print("Finished collecting good samples, shape of accepted simulations: {}".format(x.shape[0]))
        if rerun:
            print("Rerunning to collect more samples, current shape of accepted simulations: {} and rmse mean was {} and min was: {}".format(x.shape[0], rmse.mean(), rmse.min()))


    invalid_predicted = torch.cat(bad_predicted, dim=0)
    nan_predicted = torch.as_tensor([float("nan")],device='cpu')*torch.ones_like(invalid_predicted)
    good_predicted = torch.cat(new_predicted, dim=0)

    invalid_theta = torch.cat(bad_theta,dim=0).to('cpu')
    good_theta = torch.cat(new_theta, dim=0).to('cpu')
    if not oversampling_factor:
        final_theta = torch.cat((good_theta, invalid_theta[:150, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:150, :]),dim=0)
    elif int(invalid_theta.shape[0]/good_theta.shape[0])>2 and invalid_theta.shape[0] > 500:
        final_theta = torch.cat((good_theta, invalid_theta[:300, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:300, :]),dim=0)
    else:
        final_theta = torch.cat((good_theta, invalid_theta),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted),dim=0)
    return final_theta, final_x

def restricted_simulations_with_embedding(proposal, embedding, rmin, lossfn, batch_size, target, simulator, sample_size=300, oversampling_factor=None):
    """_summary_

    Args:
        proposal (_type_): _description_
        embedding (_type_): _description_
        lossfn (_type_): _description_
        batch_size (_type_): _description_
        true_x (_type_): _description_
        simulator (_type_): _description_
        sample_size (int, optional): _description_. Defaults to 300.
        oversampling_factor (_type_, optional): _description_. Defaults to None.
    """

     #calculate l2 distance between predicted and true for the first 10 bins
    rerun = True
    new_predicted = []
    new_theta = []
    bad_theta = []
    bad_predicted = []
    prev_shape = 0
    count = 0
    with torch.no_grad():
        while rerun:
            if not oversampling_factor:
                theta = proposal.sample((batch_size,)).to('cpu')
            else:
                theta = proposal.sample((batch_size,),max_sampling_batch_size=5000, oversampling_factor=oversampling_factor).to('cpu')
            predicted = simulate_in_batches(simulator, theta, num_workers=1, show_progress_bars=False)
            embedding_predicted = embedding(predicted.unsqueeze(1).to(the_device))
            norm_predicted = (embedding_predicted.squeeze(1)/embedding_predicted.squeeze(1).sum(dim=1).view(embedding_predicted.shape[0],1)).unsqueeze(1)

            loss = lossfn(norm_predicted, target.repeat(norm_predicted.shape[0],1,1))
            best_idx_unique = torch.lt(loss, rmin*torch.ones_like(loss)).to('cpu')
            temp_theta = theta[best_idx_unique]
            temp_predicted = predicted[best_idx_unique,:]

            bad_theta.append(theta[~best_idx_unique])
            bad_predicted.append(predicted[~best_idx_unique])
            # append the valid simulations
            new_predicted.append(temp_predicted)
            new_theta.append(temp_theta)
            # stop when we have enough good simulations
            x = torch.cat(new_predicted, dim=0)
            count += 1

            if x.shape[0] >= sample_size:
                rerun = False
                print("Finished collecting good samples, shape of accepted simulations: {}".format(x.shape[0]))
            if rerun and count%20==0:
                print("Rerunning to collect more samples, current shape of accepted simulations: {} and rmse mean was {} and min was: {}".format(x.shape[0], loss.mean(), loss.min()))
            if rerun and prev_shape < x.shape[0]:
                print("Added more samples, current shape of accepted simulations: {}: ".format(x.shape[0]))
                prev_shape = x.shape[0]
            if len(bad_predicted) > 300:
                bad_predicted = bad_predicted[:150]
                bad_theta = bad_theta[:150]



    invalid_predicted = torch.cat(bad_predicted, dim=0)
    nan_predicted = torch.as_tensor([float("nan")],device='cpu')*torch.ones_like(invalid_predicted)
    good_predicted = torch.cat(new_predicted, dim=0)

    invalid_theta = torch.cat(bad_theta,dim=0).to('cpu')
    good_theta = torch.cat(new_theta, dim=0).to('cpu')
    if not oversampling_factor:
        final_theta = torch.cat((good_theta, invalid_theta[:150, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:150, :]),dim=0)
    elif int(invalid_theta.shape[0]/good_theta.shape[0])>2 and invalid_theta.shape[0] > 500:
        final_theta = torch.cat((good_theta, invalid_theta[:300, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:300, :]),dim=0)
    else:
        final_theta = torch.cat((good_theta, invalid_theta),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted),dim=0)
    return final_theta, final_x


def restricted_simulations_with_embedding_find_min(proposal, embedding, lossfn, batch_size, num_rounds, target, simulator):
    """_summary_

    Args:
        proposal (_type_): _description_
        embedding (_type_): _description_
        lossfn (_type_): _description_
        batch_size (_type_): _description_
        true_x (_type_): _description_
        simulator (_type_): _description_
        sample_size (int, optional): _description_. Defaults to 300.
        oversampling_factor (_type_, optional): _description_. Defaults to None.
    """

     #calculate l2 distance between predicted and true for the first 10 bins
    min_loss = []
    mean_loss = []
    max_loss = []

    for i in tqdm(range(0, num_rounds)):

        theta = proposal.sample((batch_size,))
        predicted = simulate_in_batches(simulator, theta, num_workers=1, show_progress_bars=False) # skpping first bin

        with torch.no_grad():
            embedding_predicted = embedding(predicted.unsqueeze(1).to(the_device))
            norm_predicted = (embedding_predicted.squeeze(1)/embedding_predicted.squeeze(1).sum(dim=1).view(embedding_predicted.shape[0],1)).unsqueeze(1)
            loss = lossfn(norm_predicted, target.repeat(norm_predicted.shape[0],1,1))
        min_loss.append(loss.min().to('cpu'))
        mean_loss.append(loss.mean().to('cpu'))
        max_loss.append(loss.max().to('cpu'))

    return min_loss, mean_loss, max_loss


def main(argv):
    create_global_variables()
    set_reproducable_seed(FLAGS.seed)

    # sets up cached simulated data to read from hdf5 file
    get_sim_datafrom_hdf5('chr10_sim_genome_wide_mut_sfs.h5')
    print("Finished Loading Dataset")


    high_param = 0.0 * torch.ones(1, device=the_device)
    low_param = -8*torch.ones(1, device=the_device)
    simple_prior = torch.distributions.Uniform(low=low_param, high=high_param)
    batch_size=20

    ind_prior = MultipleIndependent(
    [
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        torch.distributions.Uniform(low=low_param, high=high_param),
        #torch.distributions.Uniform(low=torch.zeros(1, device=the_device), high=1*torch.ones(1,device=the_device)),
    ],
    validate_args=False,)
    # Set up prior and simulator for SBI
    prior, num_parameters, prior_returns_numpy = process_prior(ind_prior)
    proposal = prior

    simulator = process_simulator(aggregated_generate_sim_data, prior, prior_returns_numpy)
    #simulator = process_simulator(generate_sim_data_only_one, prior, prior_returns_numpy)


    print("Creating embedding network")

    embedding_net = SummaryNet(sample_size*2-1, [32, 32, 16]).to(the_device)


    true_x = (load_true_data('emperical_lof_sfs_nfe.npy', 0)).unsqueeze(0)
    embedding_true_x = embedding_net(true_x.unsqueeze(0).to(the_device)) # shoudl be of shape batch size x 1 x sample-sze
    embedding_true_x_norm = (embedding_true_x.squeeze(1)/embedding_true_x.squeeze(1).sum()).unsqueeze(1)
    print("True data shape (should be the same as the sample size): {} {}".format(true_x.shape[0], sample_size*2-1))


    print("Starting to Train")

    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.99)
    estimator_round = 200
    load_classifier=False

    num_rounds=1000
    nmin, nmean, nmax = restricted_simulations_with_embedding_find_min(proposal, embedding_net, sinkhorn,
                                                             batch_size, num_rounds, embedding_true_x_norm, simulator)
    print("lowest min {}, max min {}, mean min {}".format(np.min(nmin), np.max(nmin), np.mean(nmin)))
    print("mean loss {}".format(np.mean(nmean)))
    print("max mean loss {}".format(np.mean(nmax)))
    # First created restricted estimator

    rmin = np.median(nmin) # get the min from initial simulations
    #rmin=1.76

    restriction_estimator = RestrictionEstimator(prior=ind_prior, decision_criterion="nan", device=the_device)
    if load_classifier:
        restriction_estimator = torch.load('nfe_restriction_classifier_lof_embedding_genome_wide.pkl')
        proposal = restriction_estimator.restrict_prior(allowed_false_negatives=0.0)


    else:
        for i in tqdm(range(0,estimator_round)):
            if i == 0:
                theta, x = restricted_simulations_with_embedding(proposal, embedding_net, rmin, sinkhorn, batch_size, embedding_true_x_norm, simulator, sample_size=50)
            else:
                theta, x = restricted_simulations_with_embedding(proposal, embedding_net, rmin, sinkhorn, batch_size, embedding_true_x_norm, simulator, sample_size=150, oversampling_factor=1024)
            restriction_estimator.append_simulations(theta, x, data_device='cpu')
            if (i < estimator_round - 1):
                # training not needed in last round because classifier will not be used anymore.
                restriction_estimator.train(loss_importance_weights=True,training_batch_size=20)
                restriction_estimator._x_roundwise = []
            if i % 20 == 0:
                theta_test = proposal.sample((1000,))
                print("theta test mean: {}.".format(theta_test.cpu().mean()))
                del theta_test
            proposal = restriction_estimator.restrict_prior(allowed_false_negatives=0.0)
        restriction_estimator._x_roundwise = None # too save memory
        save_estimator = deepcopy(restriction_estimator)
        save_estimator._build_nn=None
        torch.save(save_estimator, 'nfe_restriction_classifier_mis_embedding_genome_wide.pkl')
        del save_estimator

    print("Finsished")


if __name__ == '__main__':
    app.run(main)
