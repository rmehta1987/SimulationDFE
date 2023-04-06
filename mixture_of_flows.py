import torch
from torch import nn
from torch.distributions import Independent, Categorical, Distribution, biject_to
from torch.distributions.constraints import Constraint, real

import numpy as np
import warnings
from typing import Optional, Iterable

from pyro.nn import AutoRegressiveNN, DenseNN
from pyro.distributions import transforms
from pyro.distributions.transforms import ComposeTransform


from first_second_order_helpers import jacobian_in_batch
from sbi.samplers.vi.vi_pyro_flows import AffineTransform, LowerCholeskyAffine, build_flow

class StackedAutoRegressiveNN(nn.Module):
    """This is implements several independent autoregressive networks, which is usefull
        for mixture distributions.
        Args:
            num_nn: Number of independent networks
            input_dim: Input dimension of each network, the total dimension of this
            network will be (num_nn, input_dim)
            hidden_dim: Hidden dimension of the autoregressive neural network
            param_dim: The shape and the number of paramters for each network. For
            Affine Autoregressive flow [1,1] is default as we want to estimate one mean
            and one scale.
    """

    def __init__(
        self,
        num_nn: int,
        input_dim: int,
        hidden_dim: Iterable = None,
        param_dims: Iterable = [1, 1],
        **kwargs,
    ):
        super().__init__()
        self.num_nn = num_nn
        self.input_dim = input_dim
        self.param_dims = param_dims
        self.nets = []
        self._kwargs = kwargs
        self.permutation = []
        if hidden_dim is None:
            self.hidden_dim = [input_dim * 5 + 10]
        else:
            self.hidden_dim = hidden_dim
        for i in range(num_nn):
            net = AutoRegressiveNN(
                input_dim,
                hidden_dims=self.hidden_dim,
                param_dims=param_dims,
                permutation=torch.arange(input_dim),  # This is important for imp. grads
                **kwargs,
            )
            self.add_module("AutoRegressiveNN" + str(i), net)
            self.nets.append(net)
            self.permutation = list(net.permutation)

    def forward(self, x):
        """ Forward pass through each network and stack it """
        x = x.reshape(-1, self.num_nn, self.input_dim)
        out = list(zip(*[net(x[:, i]) for i, net in enumerate(self.nets)]))
        result = [
            torch.hstack(o)
            .reshape(-1, self.num_nn, self.param_dims[i], self.input_dim)
            .squeeze()
            for i, o in enumerate(out)
        ]
        return tuple(result)


class StackedDenseNN(nn.Module):
    """This is implements several independent dense networks, which is usefull
        for mixture distributions of coupling flows.
        Args:
            num_nn: Number of independent networks
            split_dim: Input dimension of each network, which corresponds to the split
            dimension in coupling flows.
            hidden_dim: Hidden dimension of the autoregressive neural network
            param_dim: The shape and the number of paramters for each network. For
            Affine flow [1,1] is default as we want to estimate one mean
            and one scale.
    """

    def __init__(
        self,
        num_nn: int,
        split_dim: int,
        hidden_dim: Iterable = None,
        param_dims: Iterable = [1, 1],
        **kwargs,
    ):
        super().__init__()
        self.num_nn = num_nn
        self.split_dim = split_dim
        self.param_dims = param_dims
        self.nets = []
        self._kwargs = kwargs
        if hidden_dim is None:
            self.hidden_dim = [split_dim * 5 + 10]
        else:
            self.hidden_dim = hidden_dim
        for i in range(num_nn):
            net = DenseNN(
                split_dim, hidden_dims=self.hidden_dim, param_dims=param_dims, **kwargs
            )
            self.add_module("DenseNN" + str(i), net)
            self.nets.append(net)

    def forward(self, x):
        """ Forward pass through each network and stack it """
        x = x.reshape(-1, self.num_nn, self.split_dim)
        out = list(zip(*[net(x[:, i]) for i, net in enumerate(self.nets)]))
        result = [
            torch.hstack(o)
            .reshape(-1, self.num_nn, self.param_dims[i], self.split_dim)
            .squeeze(-1)
            .squeeze(0)
            for i, o in enumerate(out)
        ]
        return tuple(result)


class MixtureSameTransform(torch.distributions.MixtureSameFamily):
    """Trainable MixtureSameFamily using transformed distributions.  The component
    distribution should be of tpye TransformedDistribution!
    
    We implement rsample for component distributions that satisfy the autoregressive
    property. If your are not sure if your model is correct use "check_rsample" method
    """

    def parameters(self):
        """ Returns the learnable paramters of the model """
        if not self._mixture_distribution.logits.requires_grad:
            self._mixture_distribution.logits.requires_grad_(True)
        yield self._mixture_distribution.logits
        if hasattr(self._component_distribution, "parameters"):
            yield from self._component_distribution.parameters()
        elif hasattr(self._component_distribution, "transforms"):
            for t in self._component_distribution.transforms:
                if isinstance(t, nn.Module) or hasattr(t, "parameters"):
                    for para in t.parameters():
                        yield para
        else:
            pass

    def modules(self):
        """ Returns the modules of the model """
        if hasattr(self._component_distribution, "modules"):
            yield from self._component_distribution.modules()
        elif hasattr(self._component_distribution, "transforms"):
            for t in self._component_distribution.transforms:
                if isinstance(t, nn.Module):
                    yield t
        else:
            pass

    def clear_cache(self):
        if hasattr(self._component_distribution, "clear_cache"):
            self._component_distribution.clear_cache()

    def conditional_logprobs(self, x):
        """ Logprobs for each component and dimension."""
        x_pad = self._pad(x)
        link_transform = self._component_distribution.transforms[-1]
        transforms = self._component_distribution.transforms[:-1]
        x_delinked = link_transform.inv(x_pad)
        x = x_delinked
        eps = torch.zeros_like(x)
        jac = torch.zeros_like(x)
        for t in reversed(transforms):
            eps = t.inv(x)
            jac += t.log_abs_jacobian_diag(eps, x)
            x = eps

        base_dist = self._component_distribution.base_dist
        if isinstance(base_dist, Independent):
            log_prob = (
                base_dist.base_dist.log_prob(eps)
                - jac
                - link_transform.log_abs_det_jacobian(x_delinked, x_pad).squeeze()
            )
        else:
            log_prob = (
                base_dist.log_prob(eps)
                - jac
                - link_transform.log_abs_det_jacobian(x_delinked, x_pad).squeeze()
            )

        return log_prob

    def _pad(self, x):
        """ Pads the input, by repeating in "_num_component" times """
        x = x.reshape(-1, self._event_shape[0])
        return x.unsqueeze(1).repeat(1, self._num_component, 1)

    def conditional_cdf(self, x):
        """ Cdfs for each component and dimension. """
        x_pad = self._pad(x)
        transform = ComposeTransform(self._component_distribution.transforms)
        eps = transform.inv(x_pad)
        base_dist = self._component_distribution.base_dist
        if isinstance(base_dist, Independent):
            cdf = base_dist.base_dist.cdf(eps)
        else:
            cdf = base_dist.cdf(eps)
        return cdf

    def standardize(self, x):
        """ This transform converts samples from the distributions to Unif[0,1]
        samples. This works only if the autoregressive property holds."""
        log_prob_x = self.conditional_logprobs(x)
        cdf_x = self.conditional_cdf(x)

        cum_sum_logq_k = log_prob_x.cumsum(2).roll(1, 2)
        cum_sum_logq_k[:, :, 0] = 0