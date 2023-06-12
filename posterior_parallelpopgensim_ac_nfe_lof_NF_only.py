import torch
from torch import nn, optim
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
from torch.distributions import Distribution, Independent, Normal, Categorical, biject_to
from torch.distributions.transforms import ComposeTransform, IndependentTransform
from torch.nn import Module
from pyro.distributions import transforms
from pyro.nn import AutoRegressiveNN, DenseNN


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

class DFEmodule(nn.Module):
    def __init__(self, dist, arg_constraints, num_transforms, **params):
        super(DFEmodule, self).__init__()
        self.dist_class = dist
        self.arg_constraints = arg_constraints.support
        self.num_transforms = num_transforms
        self.event_shape = dist.event_shape
        #for k, v in params.items():
        #    self.register_parameter(k, nn.Parameter(v))
        self.build_flow()

    def get_dist(self):
        constrained_params = dict(self.get_constrained_params())
        return self.dist_class(**constrained_params)

    def log_prob(self, sample):
        return self.flow_dist.log_prob(sample)

    def forward(self):
        return self.flow_dist.rsample()

    def rsample(self, samples):
        return self.flow_dist.rsample((samples,))

    def loss(self):
        samples = self.q.rsample((self.n_particles,))
        logq = self.q.log_prob(samples)
        return logq
    
    def build_flow(self):
        
        flows = []
        input_dim = self.event_shape[0]
        split_dim = int(input_dim/2)
        count_bins = 8
        hidden_dims = [5 * input_dim + 30, 5 * input_dim + 30]
        param_dims = [(input_dim - split_dim) * count_bins,
                      (input_dim - split_dim) * count_bins,
                      (input_dim - split_dim) * (count_bins - 1),
                      (input_dim - split_dim) * count_bins]
        hypernet = DenseNN(split_dim, hidden_dims, param_dims)
        flows.append(transforms.SplineCoupling(input_dim, split_dim, hypernet))
        for i in range(self.num_transforms):
            flows.append(transforms.SplineCoupling(input_dim, split_dim, hypernet))
        if i < self.num_transforms - 1:
            permutation = torch.randperm(input_dim)
            flows.append(transforms.Permute(permutation))
        if i < self.num_transforms - 1:
            bn = transforms.BatchNorm(input_dim)
            flows.append(bn)
        flows.append(biject_to(self.arg_constraints))
        self.flow_dist = torch.distributions.TransformedDistribution(self.dist_class, flows)
        
    def get_constrained_params(self):
        for name, param in self.named_parameters():
            yield name, biject_to(self.arg_constraints)

class simpleMLP(nn.Module):
    def __init__(self, shape, hidden_features=100, dropout_prob = 0.5):
        super(simpleMLP, self).__init__()

        self.neural_net = nn.Sequential(
        nn.Linear(shape, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, hidden_features),
        nn.BatchNorm1d(hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, 2),)
    
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.neural_net(x)

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
    scaling_theta = 10**prior[-1].cpu().numpy()
    mis_id=0
    for a_prior in gammas:
        _, idx = loaded_tree.query(a_prior, k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
        fs = loaded_file[loaded_file_keys[idx[0]]][:]
        fs = (1 - mis_id)*fs + mis_id * fs[::-1]
        #fs = fs*.01552243512  # scale to lof theta
        fs = fs*scaling_theta  # scale to lof theta
        data += fs
    data /= theprior.shape[0]
    return torch.exp(torch.log(torch.nn.functional.relu(torch.tensor(data)+1)).type(torch.float32))

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

def restricted_simulations_with_embedding(proposal, scaling, embedding, rmin, lossfn, l1loss, batch_size, target, simulator, sample_size=50):
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
            theta = proposal.rsample(batch_size).detach().to('cpu')
            scalar = scaling.sample((batch_size,1))
            theta = torch.cat((theta, scalar),dim=1)
            if theta is not None:
                predicted = simulate_in_batches(simulator, theta, num_workers=1, show_progress_bars=False)
                embedding_predicted = embedding(predicted.unsqueeze(1).to(the_device))
                norm_predicted = (embedding_predicted.squeeze(1)/embedding_predicted.squeeze(1).sum(dim=1).view(embedding_predicted.shape[0],1)).unsqueeze(1)

                loss = lossfn(norm_predicted, target.repeat(norm_predicted.shape[0],1,1))
                loss2 = l1loss(norm_predicted, target.repeat(norm_predicted.shape[0],1,1))
                loss2 = loss2.squeeze().mean(dim=1).min()
                best_idx_unique = torch.le(loss+loss2, rmin*torch.ones_like(loss)).to('cpu')
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
            else:
                print("Rejected sampler could not find enough samples within enough time, rerunning")
                rerun = False

            
    invalid_predicted = torch.cat(bad_predicted, dim=0)
    nan_predicted = torch.as_tensor([float("nan")],device='cpu')*torch.ones_like(invalid_predicted)
    good_predicted = torch.cat(new_predicted, dim=0)

    invalid_theta = torch.cat(bad_theta,dim=0).to('cpu')
    good_theta = torch.cat(new_theta, dim=0).to('cpu')
    if invalid_theta.shape[0] > good_theta.shape[0]:
        shape_param = good_theta.shape[0]
        final_theta = torch.cat((good_theta, invalid_theta[:shape_param, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:shape_param, :]),dim=0)
    elif int(invalid_theta.shape[0]/good_theta.shape[0])>2 and invalid_theta.shape[0] > 500:
        final_theta = torch.cat((good_theta, invalid_theta[:300, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:300, :]),dim=0)
    else:
        final_theta = torch.cat((good_theta, invalid_theta),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted),dim=0)
    return final_theta, final_x


def restricted_simulations_with_no_embedding(proposal, scaling, rmin, lossfn, l1loss, batch_size, target, simulator, sample_size=50):
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
            theta = proposal.rsample(batch_size).detach().to('cpu')
            scalar = scaling.sample((batch_size,1))
            theta = torch.cat((theta, scalar),dim=1)
            if theta is not None:
                predicted = simulate_in_batches(simulator, theta, num_workers=1, show_progress_bars=False)
                norm_predicted = (predicted.squeeze(1)/predicted.squeeze(1).sum(dim=1).view(predicted.shape[0],1)).unsqueeze(1)

                loss = lossfn(norm_predicted, target.repeat(norm_predicted.shape[0],1,1))
                #loss2 = l1loss(norm_predicted, target.repeat(norm_predicted.shape[0],1,1))
                #loss2 = loss2.squeeze().mean(dim=1).min()
                best_idx_unique = torch.le(loss, rmin*torch.ones_like(loss)).to('cpu')
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
            else:
                print("Rejected sampler could not find enough samples within enough time, rerunning")
                rerun = False

            
    invalid_predicted = torch.cat(bad_predicted, dim=0)
    nan_predicted = torch.as_tensor([float("nan")],device='cpu')*torch.ones_like(invalid_predicted)
    good_predicted = torch.cat(new_predicted, dim=0)

    invalid_theta = torch.cat(bad_theta,dim=0).to('cpu')
    good_theta = torch.cat(new_theta, dim=0).to('cpu')
    if invalid_theta.shape[0] > good_theta.shape[0]:
        shape_param = good_theta.shape[0]
        final_theta = torch.cat((good_theta, invalid_theta[:shape_param, :]),dim=0)
        final_x= torch.cat((good_predicted, nan_predicted[:shape_param, :]),dim=0)
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
    # sets up cached simulated data to read from hdf5 file
    get_sim_datafrom_hdf5('chr10_sim_genome_wide_mut_sfs.h5')
    print("Finished Loading Dataset")


    print("Creating embedding network")
    embedding_net = SummaryNet(sample_size*2-1, [32, 32, 16]).to(the_device)


    true_x = (load_true_data('emperical_lof_sfs_nfe.npy', 0)).unsqueeze(0)
    true_x_norm = (true_x/true_x.sum())
    embedding_true_x = embedding_net(true_x.unsqueeze(0).to(the_device)) # shoudl be of shape batch size x 1 x sample-sze
    embedding_true_x_norm = (embedding_true_x.squeeze(1)/embedding_true_x.squeeze(1).sum()).unsqueeze(1)
    print("True data shape (should be the same as the sample size): {} {}".format(true_x.shape[0], sample_size*2-1))
    prior = torch.distributions.Uniform(-7.0, 0)
    scalar_prior = torch.distributions.Uniform(-3.0, 0)
    simulator = process_simulator(aggregated_generate_sim_data, prior, False)
    #scaling_parameter = nn.Parameter()

    print("Starting to Train")

    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.99)
    l1loss = nn.L1Loss(reduction='none')
    num_rounds=1000

    input_dim = 10

    bdist = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(input_dim), torch.ones(input_dim)),1)
    constraints = torch.distributions.Uniform(-7.0, 0)



    testnf = DFEmodule(bdist, constraints, 1)
    theta = testnf.rsample(100)
    theta = theta.detach()
    scaling = scalar_prior.sample((100,1))
    theta = torch.cat((theta, scaling),dim=1)
    predicted = simulate_in_batches(simulator, theta, num_workers=1, show_progress_bars=False)
    predicted_norm = (predicted/predicted.sum(dim=1).view(predicted.shape[0],1)).unsqueeze(1)


    embedding_predicted = embedding_net(predicted.unsqueeze(1).to(the_device))
    norm_predicted = (embedding_predicted.squeeze(1)/embedding_predicted.squeeze(1).sum(dim=1).view(embedding_predicted.shape[0],1)).unsqueeze(1)

    loss = torch.min(sinkhorn(norm_predicted, embedding_true_x_norm.repeat(norm_predicted.shape[0],1,1))) # Find the minimum loss
    loss_2_sink = torch.min(sinkhorn(predicted_norm, true_x_norm.repeat(predicted_norm.shape[0],1,1)))
    loss_l1_2 = l1loss(predicted, true_x.repeat(norm_predicted.shape[0],1,1))
    loss_l1_2 = loss_l1_2.squeeze().mean(dim=1).min()
    loss_l1 = l1loss(norm_predicted, embedding_true_x_norm.repeat(norm_predicted.shape[0],1,1))
    loss_l1 = loss_l1.squeeze().mean(dim=1).min()

    data_theta, data_x = restricted_simulations_with_no_embedding(testnf, scalar_prior, loss_2_sink, sinkhorn, l1loss, 100, true_x_norm, simulator, 5)
    data_theta = data_theta[:,:-1]
    data_scalar = data_theta[:,1]
    
    labels = 1-torch.isnan(data_x[:,1]).type(torch.int)

    classifier_gamma = simpleMLP(data_theta.shape[0])
    classifier_theta = simpleMLP(data_scalar.shape[0])

    optimizer = optim.Adam(list(classifier_gamma.parameters(), classifier_theta.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction="none")
    outputs_gamma = classifier_gamma(data_theta)
    cl_loss_gamma = criterion(outputs_gamma, labels).mean()
    outputs_theta = classifier_gamma(data_scalar)
    cl_loss_scalar = criterion(outputs_theta, labels).mean()

    loss = cl_loss_gamma + cl_loss_scalar
    loss.backward()

    print(data_theta.shape)




if __name__ == '__main__':
    app.run(main)
