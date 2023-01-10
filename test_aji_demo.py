import moments
import numpy as np
from matplotlib import pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR) # See: https://github.com/matplotlib/matplotlib/issues/14523
import pdb

import demes

'''

N0 = 13945 # initial effective pop size
l = [
  [2*N0],
  [2*3874 * np.exp(0.015 * t) for t in range(89342)],
  [2*1000 * np.exp(0.05 * t) for t in range(87)], # using 1% growth
  [(2*1000 * np.exp(0.05 * 87)) * np.exp(0.3 * t) for t in range(1, 13)], # using of 20% growth
]
flat_list = [item for sublist in l for item in sublist]
Nc = np.array(flat_list)/(2*N0) # moments needs pop size as floats scaled by initial effective pop size
nu_func = lambda t: [Nc[int(t*2*N0)]] # creating a function to return *scaled* Ne at each time point/gen
fs = moments.LinearSystem_1D.steady_state_1D(16734, gamma=-20, theta=1.0)
fs = moments.Spectrum(fs)
fs.integrate(nu_func, 2.8, gamma=-20, dt_fac=0.002, theta=1.0) # 2.8 is the total number of gens in pop size units

from moments.Numerics import compute_N_effective


#pdb.set_trace()
print(flat_list[-1])

#Neff = compute_N_effective(nu_func, 5.5*0.5, 0.5 * (5.5 + 0.02))
fig = moments.Plotting.plot_1d_fs(fs)
plt.savefig('finish_demo.png')

'''


import demes

ooa_graph = demes.load("/home/rahul/PopGen/SimulationSFS/aji_deme.yaml")
isinstance(ooa_graph, demes.Graph)

moment_data =  moments.Spectrum.from_demes(ooa_graph, sampled_demes=["AJI"], sample_sizes=[100], gamma=-10.0)
actual_fs = moment_data.compressed()  
print(actual_fs[:10])