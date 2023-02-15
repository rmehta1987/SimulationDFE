import numpy as np



a_path = 'emperical_missense_variant_sfs.csv'
ac_count = np.loadtxt(a_path, delimiter=",", dtype=object, skiprows=1)
ac_count = ac_count[:,0].astype(float) # first column is the counts of alleles and converts to float
sample_size=56500
thebins = np.arange(1,sample_size*2+1) 
# Get the histogram
sfs, bins = np.histogram(ac_count, bins=thebins)  
print(sfs.shape[0])
assert sfs.shape[0] == sample_size*2-1, "True Sample Size must be the same dimensions as the Site Frequency Spectrum for the true sample size, SFS shape: {} and sample shape: {}".format(sfs.shape[0], sample_size)
np.save('wrong_sample_size_emperical_missense_sfs_nfe_ac_saved.pkl', sfs)
