import numpy as np
import moments

def generate_momments(prior: float, sample_size):
    
    opt_params = [2.21531687, 5.29769918, 0.55450117, 0.04088086]
    theta_mis = 15583.437265450002
    theta_lof = 1164.3148344084038
    rerun = True
    ns_sim = 100
    h=0.5
    projected_sample_size = sample_size*2
    gamma = -1*10**(prior)
    p_misid = 0 #.0021 # lof missid

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
            print("rerunning")
        else:
            rerun = False
        fs2 = fs.project([projected_sample_size]).compressed()*theta_lof
        fs2 = (1 - p_misid) * fs2 + p_misid * fs2[::-1]
        
    return fs2

prior1 = -2.2848094936497882
prior2 = -2.2012067068902295
fs1 = generate_momments(prior1, 85)
#fs2 = generate_momments(prior1+np.abs(0.00036668), 85)
fs2 = generate_momments(prior2, 85)
print(fs1-fs2)