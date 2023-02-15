import demes
import demesdraw
import matplotlib.pyplot as plt
import numpy as np
import yaml # forward in time

b = demes.Builder(
    description="Ipsita Mutation Saturation Demographic One Population",
    doi=["10.1371/journal.pgen.1000695"],
    time_units="years",
    generation_time=25,
)

# From https://github.com/agarwal-i/cpg_saturation/blob/main/msprime/SDeur.py
Ne = [14448,14068,14068,14464,14464,15208,15208,16256,16256,17618,17618,19347,19347,21534,21534,24236,24236,27367,27367,30416,30416,32060,32060,31284,29404,26686,23261,18990,16490,16490,12958,12958,9827,9827,7477,7477,5791,5791,4670,4670,3841,3841,3372,3372,3287,3359,3570,4095,4713,5661,7540,11375,14310,13292,14522,613285] #613285
T = [55940,51395,47457,43984,40877,38067,35501,33141,30956,28922,27018,25231,23545,21951,20439,19000,17628,16318,15063,13859,12702,11590,10517,9482,8483,7516,6580,5672,5520,5156,4817,4500,4203,3922,3656,3404,3165,2936,2718,2509,2308,2116,1930,1752,1579,1413,1252,1096,945,798,656,517,383,252,124,50]
Ne_unique = np.take(Ne, np.sort(np.unique(Ne, return_index=True)[1]))
T_unique = np.take(T, np.sort(np.unique(Ne, return_index=True)[1]))

# Forward in time
epochs = []
demolist = [[]]*len(T_unique)
#demodict = {"Description": "A single population model with epochs of exponential growth and decay.",  "doi:" : "https://doi.org/10.1038/ng.3015", "time_units: ": "generations", "demes: ": } 

# Reverse in time loop
#for a_t, ne in zip(T_, Ne):
#    epochs.append(dict(end_time=a_t, start_size=ne))
#epochs.append(dict(end_time=0, start_size=Ne[-1]))

'''
# Forward in time loop
count = 0
for a_t, ne in zip(T, Ne):
    if count == 0:
        fwd_time = 0
        demolist[count] = [0, ne]
        count += 1
        epochs.append(dict(end_time=0, start_size=ne))
    else:
        fwd_time = T[count - 1] - a_t + fwd_time
        epochs.append(dict(end_time=fwd_time, start_size=ne))
        demolist[count] = [fwd_time, ne]
        count += 1 
np.savetxt('ispital_generation_demo.txt', np.asarray(demolist), fmt='%d')
'''
count = 0
for a_t, ne in zip(T_unique, Ne_unique):
    if count == 0:
        fwd_time = 0
        demolist[count] = [0, ne]
        count += 1
        epochs.append(dict(end_time=0, start_size=ne))
    else:
        fwd_time = T_unique[count - 1] - a_t + fwd_time
        epochs.append(dict(end_time=fwd_time, start_size=ne))
        demolist[count] = [fwd_time, ne]
        count += 1 
#np.savetxt('ispital_smaller_generation_demo.txt', np.asarray(demolist), fmt='%d')

# Forward in time-smaller epochs

# create variable names 
var_list = []
#typedef Sim_Model::demography_piecewise<epoch_26_to_27, Sim_Model::demography_constant> epoch_27_to_28;

for i in range(1, len(T_unique)):
    #var_list.append(f"typedef Sim_Model::demography_piecewise<epoch_{i-1}_to_{i}, Sim_Model::demography_constant> epoch_{i}_to_{i+1};")
    var_list.append(f"epoch_{i}_to_{i+1} epoch_{i}(epoch_{i-1}, pop_history[{i+1}], inflection_points[{i+1}]);")
    #var_list.append(f"inflection_points.push_back({demolist[i][0]});")
    #var_list.append(f"pop_history.push_back({demolist[i][1]});")
np.savetxt('var_names_ispita_demo.txt', np.asarray(var_list), fmt='%s')


#epochs.append(dict(start_size=ne))
#demolist_pop_time = np.asarray([Ne, T]).T
#demolist_pop_time = np.append(demolist_pop_time, [[Ne[-1],0]], axis=0)

#debugging purposes
#print(demolist)


#b.add_deme(name="Schiffels_Durbin", epochs=epochs)

#sd_graph = b.resolve()
#ax = demesdraw.tubes(sd_graph, log_time=True)
#ax.figure.savefig('sd_graph.png')
#plt.close()

#ax = demesdraw.size_history(sd_graph, log_time=True)
#ax.figure.savefig('sd_size_history.png')
#plt.close()

