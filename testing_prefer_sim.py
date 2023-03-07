import numpy as np
import datetime
import subprocess
import timeit

path_to_prefer_sim = '/home/rahul/PopGen/PreferSim/PReFerSim/noParamPReFerSim'
sim_out = '/home/rahul/PopGen/SimulationSFS/prefersim_out/pout_{}'.format(datetime.datetime.now().strftime('%H_%M_%S'))
#demog_file = '/home/rahul/PopGen/SimulationSFS/exponential_growth_schiffels_durbin.txt'
demog_file = '/home/rahul/PopGen/SimulationSFS/sd_simple_exponential.txt'
num_workers = 2
mut_rate = 1
sample_size = 113000
def preferSim_simulation(theta):
    '''
    GSL_RNG_SEED=1 GSL_RNG_TYPE=mrg ./noParamPReFerSim .04 exponential_growth_schiffels_durbin.txt nfeTest/noParameterv3 2

    
    '''
    def execute_command(command):
        print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, errs = process.communicate()
        check_returncode = process.poll()
        if check_returncode == 0:
            print("Finished a simulation")
        else:
            raise Exception("Something went wrong: {}".format(errs))

    execute_command("GSL_RNG_SEED=1 GSL_RNG_TYPE=mrg {} {} {} {} {} {} {}".format(path_to_prefer_sim, theta, demog_file, sim_out, mut_rate, sample_size, num_workers))

st = timeit.default_timer()
theta = -0.05
preferSim_simulation(theta)
print("time: {}".format(timeit.default_timer()-st))
