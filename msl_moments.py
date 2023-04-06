import moments
import pickle
import numpy as np


opt_params = [2.21531687, 5.29769918, 0.55450117, 0.04088086]

data = pickle.load(open("msl_data.bp", "rb"))

fs_syn = data["spectra"]["syn"]
fs_mis = data["spectra"]["mis"]
fs_lof = data["spectra"]["lof"]

u_syn = data["rates"]["syn"]
u_mis = data["rates"]["mis"]
u_lof = data["rates"]["lof"]

opt_theta = 11372.91 * 4 * u_syn

theta_mis = opt_theta * u_mis / u_syn


def model_func(params, ns):
    nuA, nuF, TA, TF, p_misid = params
    fs = moments.Demographics1D.snm(ns)
    fs.integrate([nuA], TA)
    nu_func = lambda t: [nuA * np.exp(np.log(nuF / nuA) * t / TF)]
    fs.integrate(nu_func, TF)
    fs = (1 - p_misid) * fs + p_misid * fs[::-1]
    return fs


def selection_spectrum(gamma, h=0.5):
    rerun = True
    ns_sim = 100
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
    fs = fs.project(fs_syn.sample_sizes)
    return fs


samples_uniform = np.logspace(-4, 3, 61)

def generate_spectrum_cache():
    spectrum_cache = {}
    spectrum_cache[0] = selection_spectrum(0).compressed()

    #sims = np.zeros((samples_uniform.shape[0], fs_mis.compressed().shape[0]))

    for a, samples in enumerate(samples_uniform):
        spectrum_cache[samples] = selection_spectrum(samples).compressed()*theta_mis

    np.save('moments_msl_spectrum_cache', spectrum_cache, allow_pickle=True)

spectrum_cache = np.load('moments_msl_spectrum_cache.npy', allow_pickle=True).item()

import scipy.stats
theta_mis = opt_theta * u_mis / u_syn
theta_lof = opt_theta * u_lof / u_syn

dxs = ((samples_uniform - np.concatenate(([samples_uniform[0]], samples_uniform))[:-1]) / 2
    + (np.concatenate((samples_uniform, [samples_uniform[-1]]))[1:] - samples_uniform) / 2)

def dfe_func(params, ns, theta=1):
    alpha, beta, p_misid = params
    fs = spectrum_cache[0] * scipy.stats.gamma.cdf(samples_uniform[0], alpha, scale=beta) # effectively neutral
    weights = scipy.stats.gamma.pdf(samples_uniform, alpha, scale=beta)
    check = 0
    check2 = []
    for samples, dx, w in zip(samples_uniform, dxs, weights):
        fs += spectrum_cache[samples] * dx * w
        check += dx * w
        check2.append(check)
    fs = theta * fs
    print("Sum of weights: {}.".format(check))
    print("\n")
    print("List of weights: {}.".format(check2))
    return (1 - p_misid) * fs + p_misid * fs[::-1]

def model_func_missense(params, ns):
    return dfe_func(params, ns, theta=theta_mis)

def model_func_lof(params, ns):
    return dfe_func(params, ns, theta=theta_lof)

shape = 0.1596
scale =  2332.3
misid = 0.0137

opt_params_mis = [shape, scale, misid]

model_mis = model_func_missense(opt_params_mis, fs_mis.sample_sizes)
