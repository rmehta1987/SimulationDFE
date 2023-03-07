import demes
import demesdraw
import matplotlib.pyplot as plt
import numpy as np



ooa_graph = demes.load("sd_simple_exponential.yml")
generic = ooa_graph["generic"]
print("How many epochs does CEU have?", len(generic.epochs))
#Epoch(start_time=21200.0, end_time=0, start_size=1000, end_size=29725, size_function='exponential', selfing_rate=0, cloning_rate=0)

demolist_pop_time = np.zeros((len(generic.epochs),2))
all_epochs = generic.epochs
for i in range(len(all_epochs)):
    demolist_pop_time[i, 0] = all_epochs[i].start_size
    demolist_pop_time[i, 1] = all_epochs[i].end_time



np.savetxt('sd_simple_exponential.txt', demolist_pop_time, fmt='%i')
