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


samples_uniform = np.linspace(-.2, -.1, 61)

sims = np.zeros((samples_uniform.shape[0], fs_mis.compressed().shape[0]))

for a, samples in enumerate(samples_uniform):
    sims[a] = selection_spectrum(samples).compressed()*theta_mis

np.save('moments_msl_sim', sims)