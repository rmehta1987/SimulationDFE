import torch
import numpy as np

def ImportanceSamplingEstimator(num_particles, sample, target, threshold):

    
    sample_low = sample-threshold
    sample_high = sample+threshold
    proposal = torch.distributions.uniform.Uniform(sample_low, sample_high)
    prop_samps = proposal.sample((num_particles,))
    target_logprobs = target.log_prob(prop_samps)
    proposal_logprobs = proposal.log_prob(prop_samps)
    log_importance_weights = target_logprobs - proposal_logprobs

    ret = torch.mean((torch.exp(log_importance_weights)))


    return ret 

test = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([10.0]))
asamp = test.sample((1,))
forced_test = torch.tensor([0.0])

prob = ImportanceSamplingEstimator(50, forced_test, test, .0025)

print(prob)