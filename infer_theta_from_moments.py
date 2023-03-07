import moments
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523
import pdb

import demes

# GET SFS of synonymous sites
temp = np.loadtxt('/home/rahul/PopGen/EffectSizes/ac_nf_syn.vcf', dtype=object)
temp2 = temp[:,-2].astype(int) # get last column
temp2 = temp2[np.nonzero(temp2)]
sample_size = 2*56885-1
sample_size2 = 16000
print("max, should be sample size : {}".format(np.max(temp2)))
thebins = np.arange(1,sample_size+1)
temphist,bins = np.histogram(temp2, bins=thebins)  




ooa_graph = demes.load("/home/rahul/PopGen/SimulationSFS/nfe_deme.yaml")
isinstance(ooa_graph, demes.Graph)

data = moments.Spectrum(temphist)
data2 = data.project([sample_size2])
model =  moments.Spectrum.from_demes(ooa_graph, sampled_demes=["CEU"], sample_sizes=[sample_size2])

print("Number of segregating sites in data:", data.S())
print("Number of segregating sites in model:", model.S())
print("Ratio of segregating sites:", data.S() / model.S())

opt_theta = moments.Inference.optimal_sfs_scaling(model, data2)

print("Optimal theta:", opt_theta)
