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
import moments
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
import sparselinear as sl
from sortedcontainers import SortedDict
from scipy.spatial import KDTree


# Flag Parser 
from absl import app 
from absl import flags

FLAGS = flags.FLAGS
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523

# Integer Flags
flags.DEFINE_integer('sample_size', 56500, 'Diploid Population Sample Size where N is the number of diploids') # should be 55855
flags.DEFINE_integer('moments_theta', 1000, 'Scaled mutation rate')
flags.DEFINE_integer('num_hidden',256, "Number of hidden layers in normalizing flow architecture")
flags.DEFINE_integer('num_sim', 100, 'How many simulations to run')
flags.DEFINE_integer('rounds', 10, 'How many round of simulations to run, (total simulations = num_sim*rounds')
flags.DEFINE_integer('seed', 10, 'A seed to set for reproducability')
flags.DEFINE_integer('true_sample_size', 56500, 'The observed data sample size')  # should be 55855
flags.DEFINE_integer('number_of_transforms', 6, "How many normalizing flow blocks to use")
flags.DEFINE_integer('num_workers', 2, "How many workers to use for parallel simulations, be careful, can cause crashing")

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
    global moments_theta
    global prior_returns_numpy
    global default_network_type
    global num_sim
    global rounds
    global posterior_type
    global Ne
    global the_num_sites_total
    global the_mut_rate
    global pre_mut_rate
    global scaling_factor

    the_device = FLAGS.the_device
    if the_device and torch.cuda.is_available():
        print("The device is: {}".format(the_device))
    else:
        the_device = 'cpu'
        print("No CUDA device is: {}".format(the_device))

    sample_size = FLAGS.sample_size # Sample size of population, n, (56,885 is the sample size of non-Finnish European overall)
    moments_theta = FLAGS.moments_theta
    num_hidden = FLAGS.num_hidden
    number_of_transforms = FLAGS.number_of_transforms
    prior_returns_numpy = FLAGS.prior_returns_numpy # Prior needs to be in numpy format for simulator
    default_network_type = FLAGS.default_network_type
    num_sim = FLAGS.num_sim
    rounds = FLAGS.rounds
    posterior_type = FLAGS.posterior_type
    Ne = FLAGS.n_effective # effect population size
    the_num_sites_total = FLAGS.num_sites_total
    the_mut_rate = FLAGS.mut_rate
    pre_mut_rate = 4*Ne*the_mut_rate*the_num_sites_total
    scaling_factor = 1000.0



def set_reproducable_seed(the_seed: int):
    utils.sbiutils.seed_all_backends(the_seed) # set seed for reproducabilty

def get_sim_data(path_to_sim_file: str):
    """_summary_

    Args:
        path_to_sim_file (str): _description_
    """    

    global loaded_file 
    global loaded_file_keys
    global loaded_tree
    loaded_file = np.load(f"{path_to_sim_file}", allow_pickle=True).item() # Loads sorted-dictionary
    loaded_file_keys = list(loaded_file.keys())
    loaded_tree = KDTree(np.asarray(loaded_file_keys)[:,None]) # needs to have a column dimension
    
def generate_sim_data(prior: float) -> torch.float32:

    _, idx = loaded_tree.query(prior.cpu().numpy(), k=(1,)) # the k sets number of neighbors, while we only want 1, we need to make sure it returns an array that can be indexed
    data = loaded_file[loaded_file_keys[idx[0]]]

    return torch.poisson(torch.tensor(data, device=the_device)).type(torch.float32)


def true_sqldata_to_numpy(a_path: str):
    """_summary_

    Args:
        a_path (str): _description_
    """    
    ac_count = np.loadtxt(a_path, delimiter=",", dtype=object, skiprows=1)
    ac_count = ac_count[:,0].astype(float) # first column is the counts of alleles and converts to float

    thebins = np.arange(1,sample_size*2-1) 
    # Get the histogram
    sfs, bins = np.histogram(ac_count, bins=thebins)  
    assert sfs.shape[0]+1 == sample_size*2-1, "True Sample Size must be the same dimensions as the Site Frequency Spectrum for the true sample size, SFS shape: {} and sample shape: {}".format(sfs.shape[0], sample_size)
    np.save('emperical_missense_sfs_nfe_ac_saved.pkl', sfs)

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


class SummaryNet(nn.Module):
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.linear1 = sl.SparseLinear(self.sample_size, int(self.sample_size / 400))
        self.linear2 = sl.SparseLinear(int(self.sample_size / 400), int(self.sample_size / 400))
        self.linear3 = sl.SparseLinear(int(self.sample_size / 400), int(self.sample_size / 600))
        self.linear4 = sl.SparseLinear(int(self.sample_size / 600), int(self.sample_size / 800))
        self.linear5 = sl.SparseLinear(int(self.sample_size / 800), int(self.sample_size / 1200))

        self.model = nn.Sequential(self.linear1, nn.SiLU(), self.linear2, nn.SiLU(), self.linear3, nn.SiLU(), self.linear4, nn.SiLU(), self.linear5)

    def forward(self, x):
        
        x=self.model(x)
        return x


class SummaryNet2(nn.Module):
    def __init__(self, sample_size):
        super().__init__()
        self.sample_size = sample_size
        self.linear1 = nn.Linear(self.sample_size, int(self.sample_size / 400)) # 283
        self.linear2 = nn.Linear(int(self.sample_size / 400), int(self.sample_size / 400)) # 283
        self.linear3 = nn.Linear(int(self.sample_size / 400), int(self.sample_size / 600)) # 188
        self.linear4 = nn.Linear(int(self.sample_size / 600), int(self.sample_size / 800)) # 141
        self.linear5 = nn.Linear(int(self.sample_size / 800), int(self.sample_size / 1200)) # 94

        self.model = nn.Sequential(self.linear1, nn.SiLU(), self.linear2, nn.SiLU(), self.linear3, nn.SiLU(), self.linear4, nn.SiLU(), self.linear5)

    def forward(self, x):
        
        x=self.model(x)
        return x



@atexit.register
def finished_table_building():
    print("\n ****************************************** Completely finished experiment ****************************************** ")

def main(argv):

    create_global_variables()
    set_reproducable_seed(FLAGS.seed)
    
    box_uniform_prior = utils.BoxUniform(low=-0.015 * torch.ones(1, device=the_device), high=-1e-8*torch.ones(1,device=the_device),device=the_device)
    # Set up prior and simulator for SBI
    prior, num_parameters, prior_returns_numpy = process_prior(box_uniform_prior)

    simulator = process_simulator(generate_sim_data, prior, prior_returns_numpy)


    print("Creating embedding network")
    embedding_net = SummaryNet(sample_size*2-1).to(the_device)
    #embedding_net = SummaryNet2(sample_size*2-1).to(the_device)
    
    print("Finished creating embedding network")
    # First learn posterior
    print("Setting up posteriors")
    #density_estimator_function = posterior_nn(model="nsf", embedding_net=embedding_net, hidden_features=num_hidden, num_transforms=number_of_transforms)
    #density_estimator_function = posterior_nn(model="nsf", hidden_features=num_hidden, num_transforms=number_of_transforms)
    density_estimator_function = posterior_nn(model="maf", embedding_net=embedding_net, hidden_features=num_hidden, num_transforms=number_of_transforms)

    infer_posterior = SNPE(prior, show_progress_bars=True, device=the_device, density_estimator=density_estimator_function)

    #posterior parameters
    vi_parameters = {"q": "maf", "parameters": {"num_transforms": 3, "hidden_dims": 256}}

    proposal = prior
    accept_reject_fns = []

    get_sim_data('/home/rahul/PopGen/SimulationSFS/sfs_missense_sim_data.npy')
    true_x = load_true_data('/home/rahul/PopGen/SimulationSFS/wrong_sample_size_emperical_missense_sfs_nfe_ac_saved.pkl.npy', 0)
    print("True data shape (should be the same as the sample size): {} {}".format(true_x.shape[0], sample_size*2-1))
    #Set path for experiments
    
    path = "Experiments/saved_posteriors_nfe_infer_missense_selection_1_{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if not (os.path.isdir(path)):
        try:
            os.mkdir(path)
        except OSError:
            print("Error in making directory")

    # Train posterior
    print("Starting to Train")
    for i in range(0,rounds):

        theta, x = simulate_for_sbi(simulator, proposal, num_sim, num_workers=1)
        theta = theta.to(the_device)
        x = x.to(the_device)
        potential_function_estimator = infer_posterior.append_simulations(theta, x, data_device=the_device ).train(force_first_round_loss=True, training_batch_size=10)

        print("\n ****************************************** Building Posterior for round {} ******************************************.\n".format(i))
        posterior = infer_posterior.build_posterior(density_estimator=potential_function_estimator, sample_with = "vi", vi_method="fKL", vi_parameters=vi_parameters)
        print("Training to emperical observation")
        # This proposal is used for Varitaionl inference posteior
        posterior_build = posterior.set_default_x(true_x).train(max_num_iters=10, quality_control=False )
        accept_reject_fn = get_density_thresholder(posterior_build, quantile=1e-4)
        proposal = RestrictedPrior(prior, accept_reject_fn, posterior_build, sample_with="sir", device=the_device)
        #proposals.append(proposal)

        ### TODO Create a Plotting UTILS function
        #temp = torch.cat((posteriors[i].sample((1000,)).unsqueeze(1), posteriors2[i].sample((1000,)).unsqueeze(1), proposals[i].sample((1000,)).unsqueeze(1)),dim=1)
        #temp = temp.squeeze()
        #print(temp.shape)
        #df = pd.DataFrame(temp.numpy(), columns=['Posterior', 'Observed Posterior', 'Proposal'])
        #sns.histplot(temp.squeeze(), label=['Posterior', 'Observed Posterior', 'Proposal'], multiple="stack")
        #plt.hist(temp.numpy(), label=['Posterior', 'Observed Posterior', 'Proposal'])
        '''sns.histplot(df, multiple='stack')
        #plt.legend(['Posterior', 'Observed Posterior', 'Proposal'])
        #plt.legend()
        plt.xlabel('Selection (2Ns)')
        plt.ylabel('Counts')
        plt.title('Histogram of Selection Coefficients Sampled from Inferred Distributions of round {}'.format(i))
        plt.tight_layout()
        plt.savefig('{}/histogram_inference_round_{}.png'.format(path, i))
        plt.close()
        
        
        sns.kdeplot(df)

        #plt.legend()
        plt.xlabel('Selection (2Ns)')
        plt.ylabel('Counts')
        plt.title('KDE plot of Selection Coefficients Sampled from Inferred Distributions of round {}'.format(i))
        plt.tight_layout()
        plt.savefig('{}/kde_inference_round_{}.png'.format(path, i))
        plt.close()'''
        #################################

        # Save posters every some rounds
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
            print("\n ****************************************** Saved Posterior for round {} ******************************************.\n".format(i))


    # Save posteriors and proposals for later use

    print("Finished Training posterior and prior")

    # used for checking plots 
    '''
    temp = posteriors[-1].sample((1000,))
    sns.kdeplot(temp.squeeze(), label="posteior", color='b')

    temp = posteriors2[-1].sample((1000,))
    sns.kdeplot(temp.squeeze(), label="observed posteior", color='g')

    temp = proposals[-1].sample((1000,))
    sns.kdeplot(temp.squeeze(), label="proposal", color='r')
    plt.legend()
    plt.savefig('test_nsf.png')
    '''
    ### Currently saving objects does not seem to work for all posteriors/samplers (SRI, restricted estimator)

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
        '''
        try:
            with open(path2, "wb") as handle:
                torch.save(proposals[i], handle)
        except:
            if i == 1:
                print("Cannot save proposal distributions")
            os.remove(path2)
        '''
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

if __name__ == '__main__':
    app.run(main)
