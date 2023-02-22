import numpy as np
from itertools import islice
from sortedcontainers import SortedDict
import re
import os

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


def main():

    missense_data_path = '/home/rahul/PopGen/SimulationSFS/ParallelPopGen_Data/sfs_sims_missense_avg_rate/'
    lof_data_path = '/home/rahul/PopGen/SimulationSFS/ParallelPopGen_Data/sfs_sims_lof_avg_rate/'
    file_name = 'sfs_lof_data'
    get_data_and_put(lof_data_path, file_name)


if __name__ == "__main__":
    main()
    

