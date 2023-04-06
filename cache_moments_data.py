import numpy as np
import moments
import h5py
import torch


def generate_moments_sim_data2(prior: float, sample_size: int):
    
    opt_params = [2.21531687, 5.29769918, 0.55450117, 0.04088086]
    theta_mis = 15583.437265450002
    theta_lof = 1164.3148344084038
    rerun = True
    ns_sim = 100
    h=0.5
    projected_sample_size = sample_size*2
    #s_prior, weights = prior[:6], prior[6:]
    #s_prior, weights = prior[:5], prior[5:]
    #s_prior, p_misid, weights = prior[:7], prior[7], prior[7:]
    gamma = prior
   
    while rerun:
        ns_sim = 2 * ns_sim
        fs = moments.LinearSystem_1D.steady_state_1D(ns_sim, gamma=gamma, h=h)
        fs = moments.Spectrum(fs)
        fs.integrate([opt_params[0]], opt_params[2], gamma=gamma, h=h)
        nu_func = lambda t: [opt_params[0] * np.exp(
            np.log(opt_params[1] / opt_params[0]) * t / opt_params[3])]
        fs.integrate(nu_func, opt_params[3], gamma=gamma, h=h)
        if abs(np.max(fs)) > 10 or np.any(np.isnan(fs)):
            # large gamma-values can require large sample sizes for stability
            rerun = True
        else:
            rerun = False

    fs2 = fs.project([projected_sample_size]).compressed()
    return fs2


def create_hdf5_dataset(file_name: str):

    gammas = np.linspace(-5, 0 , 20000)
    with h5py.File(file_name, "w") as the_file:
        for gamma in gammas:
            the_sfs = generate_moments_sim_data2(gamma, sample_size=85)
            str_gamma = '{:6f}'.format(gamma)
            the_file.create_dataset(str_gamma, data=the_sfs)
    
    print('Finished creating hdf5 dataset')

def main():


    file_name = 'moments_msl_sfs_lof_hdf5_data.h5'
    create_hdf5_dataset(file_name)


if __name__ == "__main__":
    main()