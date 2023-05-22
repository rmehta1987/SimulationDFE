import numpy as np
from itertools import islice
from sortedcontainers import SortedDict
import re
import os
import hdfdict
import h5py

def closest(sorted_dict, key):
    "Return closest key in `sorted_dict` to given `key`."
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))

def get_data_and_put(data_path: str, file_name: str,):

    # First filter out files to match specific file; ie sfs...{}.txt
    lsdirs = os.listdir(data_path)
    regex_remove_last = re.compile(r'((?!last).)*$')
    sel_list_files = list(filter(regex_remove_last.match,lsdirs))

    regex_capture_selection_files = re.compile(r'^.*[{|}].*$')
    sel_list_files = list(filter(regex_capture_selection_files.match, sel_list_files))
    regex_capture_selection_coef = re.compile(r'(?<={)(.*)(?=})')
    sel_sfs_dict = SortedDict()
    for a_file in sel_list_files:
        full_file_path = f'{data_path}{a_file}'
        sel_coef = regex_capture_selection_coef.search(a_file)[1] # gets the actual selection coefficient
        the_sfs = np.loadtxt(full_file_path, dtype=float)[1:] # first line is number of mutations
        sel_sfs_dict[sel_coef] = the_sfs

    print('Finished creating sorted-dictionary')
    np.save(file_name, sel_sfs_dict, allow_pickle=True)

def convert_dict_to_hdf5(data_path: str, file_name: str):
    thedict = np.load(data_path, allow_pickle=True).item()
    with h5py.File(file_name, "w") as the_file:
        hdfdict.dump(thedict, the_file)

def create_hdf5_dataset(data_path: str, file_name: str):

        # First filter out files to match specific file; ie sfs...{}.txt
    lsdirs = os.listdir(data_path)
    regex_remove_last = re.compile(r'((?!last).)*$')
    sel_list_files = list(filter(regex_remove_last.match,lsdirs))

    regex_capture_selection_files = re.compile(r'^.*[{|}].*$')
    sel_list_files = list(filter(regex_capture_selection_files.match, sel_list_files))
    regex_capture_selection_coef = re.compile(r'(?<={)(.*)(?=})')

    with h5py.File(file_name, "w") as the_file:
        for a_file in sel_list_files:
            full_file_path = f'{data_path}{a_file}'
            sel_coef = regex_capture_selection_coef.search(a_file)[1] # gets the actual selection coefficient
            the_sfs = np.loadtxt(full_file_path, dtype=float)[1:] # first line is number of mutations
            the_file.create_dataset(sel_coef, data=the_sfs)

    print('Finished creating hdf5 dataset')

def main():

    #missense_data_path = 'missense_sim_sfs_data/'
    #lof_data_path = '/home/rahul/PopGen/SimulationSFS/ParallelPopGen_Data/sfs_sims_lof_avg_rate/'
    #lof_data_path2 = '/home/rahul/PopGen/SimulationSFS/cur_data_made_02_22/'
    data_path = '/project/jjberg/mehta5/ParallelPopGen/examples/example_dadi/chr10_genome_wide/'
    file_name = 'chr10_sim_genome_wide_mut_sfs.h5'
    create_hdf5_dataset(data_path, file_name)


if __name__ == "__main__":
    main()


