import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nflows.flows.base import Flow
from typing import Optional, List
from torch import Tensor
from .base import Base

# Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
#  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743

def log_min_exp(a, b, epsilon=1e-8):
    """
    Source: https://github.com/jornpeters/integer_discrete_flows
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
     log(exp(a) - exp(b))
     c + log(exp(a-c) - exp(b-c))
     a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y

def log_integer_probability(x, mean, logscale):
    scale = torch.exp(logscale)

    logp = log_min_exp(
        F.logsigmoid((x + 0.5 - mean) / scale),
        F.logsigmoid((x - 0.5 - mean) / scale))

    return logp

def log_zero_poisson(x, mean):
    # zero truncated poisson distribution, https://gist.github.com/ririw/2e3a4415dc8271bd2d132c476b98b567#file-ztp-py
    p = torch.special.xlogy(x,mean) - torch.log(torch.expm1(x)) - torch.lgamma(x+1)
    
    return torch.where(1 * torch.eq(mu, 0) * torch.eq(x, 0),0,p)
    

# We need to also turn torch.round (i.e., the rounding operator) into a differentiable function.
# For this purpose, we use the rounding in the forward pass, but the original input for the backward pass.
# This is nothing else than the straight-through estimator.
class RoundStraightThrough(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input, out=None)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input



class Permute(Base):
    def __init__(self, n_channels: None, features: None):
        """Inialize a permutation layer

        Args:
            n_channels (None): Channels to permute
            features (None): Columns to permute

        Raises:
            ValueError: Only n_channels or features can be defined, not both 
        """        
        super().__init__()
        
        assert isinstance(n_channels, int) ^ isinstance(features, int), "Only n_channels or features can be defined, not both"
        if n_channels:
            permutation = np.arange(n_channels, dtype='int')
            np.random.shuffle(permutation)

            permutation_inv = np.zeros(n_channels, dtype='int')
            permutation_inv[permutation] = np.arange(n_channels, dtype='int')

            self.permutation = torch.from_numpy(permutation)
            self.permutation_inv = torch.from_numpy(permutation_inv)
        elif features:
            permutation = np.random.permutation(features)

            permutation_inv = np.zeros(features, dtype='int')
            permutation_inv[permutation] = np.arange(features, dtype='int') # changes back to original spatial ordering

            self.permutation = torch.from_numpy(permutation)
            self.permutation_inv = torch.from_numpy(permutation_inv)
        else:
            raise ValueError('Define weather to permute along channels (n_channels) or columns (featuers)')

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z = z[:, self.permutation, :, :]
        else:
            z = z[:, self.permutation_inv, :, :]

        return z, ldj

    def InversePermute(self):
        '''
            Not used?
        '''
        inv_permute = Permute(len(self.permutation))
        inv_permute.permutation = self.permutation_inv
        inv_permute.permutation_inv = self.permutation
        return inv_permute

# That's the class of the Integer Discrete Flows (IDFs).
# There are two options implemented:
# Option 1: The bipartite coupling layers as in (Hoogeboom et al., 2019).
# Option 2: A new coupling layer with 4 parts as in (Tomczak, 2020).
# Option 3: Where the architecture is IDF++ (van den Berg et al 2021)
# Option 4: Where the input is a one dimensional set of counts and not images.
# Option 5: Pass in your own architecture
# We implemnet the second option explicitely, without any loop, so that it is very clear how it works.
class IDF(Flow):
    def __init__(self, netts, num_flows: int=8, features: int=100, hidden_features: int=50, embedding_net: nn.Module = nn.Identity(), **kwards):
        super(IDF, self).__init__()
        self.device = cuda
        D = features
        M= hidden_features
        # Option 1:
        if isinstance(netts, list):
            self.t  = torch.nn.ModuleList([netts[0]() for _ in range(num_flows)])
            self.idf_git = 5
        elif netts == 1:
            nett = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                     nn.Linear(M, M), nn.LeakyReLU(),
                                     nn.Linear(M, D // 2))
            netts = [nett]   
            self.t = torch.nn.ModuleList([netts[0]() for _ in range(num_flows)])
            self.idf_git = 1
        elif netts == 2:
            self.t_a = torch.nn.ModuleList([netts[0]() for _ in range(num_flows)])
            self.t_b = torch.nn.ModuleList([netts[1]() for _ in range(num_flows)])
            self.t_c = torch.nn.ModuleList([netts[2]() for _ in range(num_flows)])
            self.t_d = torch.nn.ModuleList([netts[3]() for _ in range(num_flows)])
            self.idf_git = 2
        elif netts == 3:
            # Create shallow dense nets for IDF flows
            raise ValueError('Dense network not impelemented yet')
        elif netts == 4:
            self.idf_git = 4
            self.t = 
        else:
            raise ValueError('There are 5 options, see docstring for information')
        
        # The number of flows (i.e., invertible transformations).
        self.num_flows = num_flows
        
        # The rounding operator
        self.round = RoundStraightThrough.apply
        
         # The dimensionality of the problem.
        self.D = features
        
        # Initialization of the parameters of the base distribution.
        # Notice they are parameters, so they are trained alongside the weights of neural networks.
        self.mean = nn.Parameter(torch.zeros(1, self.D)) #mean
        self.logscale = nn.Parameter(torch.ones(1,self.D)) #log-scale
        
       
    
    # The coupling layer.
    def coupling(self, x, index, forward=True):
        
        # Option 1:
        if self.idf_git == 1:
            (xa, xb) = torch.chunk(x, 2, 1)
            
            if forward:
                yb = xb + self.round(self.t[index](xa))
            else:
                yb = xb - self.round(self.t[index](xa))
            
            return torch.cat((xa, yb), 1)
        else:
            assert self.idf_git == 4, "Have not implemented other coupling layers yet"
            
            
    
    # Similalry to RealNVP, we have also the permute layer.
    # This is now defined as a separate object since we want to store the inverse permutation for
    # every permuate layer
    '''
    def permute(self, x: Tensor):
        """Utility function. Permute x with given indices. Search tf.gather for detailed use of this function.
        Args:
            x (Tensor):The data to permute along random indicies (produced by numpy permutations)
        """
        # Note that self.D is x.shape[1], which is the number of features       
        return torch.gather(x, indices=np.random.permutation(self.D), axis=-1)
    '''
    # The main function of the IDF: forward pass from x to z.
    def forward_flow(self, x):
        # Goes from current x to base distriburtion Z
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z
    
    # The function for inverting z to x.
    def inverse(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x
    
    # The PyTorch forward function. It returns the log-probability.
    def forward(self, x, reduction='avg'):
        z = self.forward_flow(x)
        if reduction == 'sum':
            return -self.log_prior(z).sum()
        else:
            return -self.log_prior(z).mean()
    
    # The function for sampling:
    # First we sample from the base distribution.
    # Second, we invert z.
    def sample(self, batchSize, intMax=100):
        # sample z:
        z = self.prior_sample(batchSize=batchSize, D=self.D)
        #z = self.prior_sample_poisson(batchSize=batchSize, D=self.D)
        # x = f^-1(z)
        x = self.inverse(z)
        return x.view(batchSize, 1, self.D)
    
    # The function for calculating the logarithm of the base distribution.
    def log_prior(self, x):
        log_p = log_integer_probability(x, self.mean, self.logscale)
        #log_p = log_zero_poisson(x, self.mean)
        return log_p.sum(1)
    
    # A function for sampling integers from the base distribution.
    def prior_sample(self, batchSize, D=2):
        # Sample from logistic
        y = torch.rand(batchSize, self.D).to(self.device)
        # Here we use a property of the logistic distribution:
        # In order to sample from a logistic distribution, first sample y ~ Uniform[0,1].
        # Then, calculate log(y / (1.-y)), scale is with the scale, and add the mean.
        x = torch.exp(self.logscale) * torch.log(y / (1. - y)) + self.mean
        # And then round it to an integer.
        return torch.round(x)

    def prior_sample_poisson(self, batchSize, D=2):
        # Sample from logistic
        y = torch.FloatTensor(batchSize, self.D).uniform_(self.mean,1).to(self.device)
        t = -1*torch.log(y)
        x = 1 + torch.poisson(y)
        # And then round it to an integer.
        #return torch.round(x)
        return x