#!/bin/bash

#SBATCH --job-name=posterioir
#SBATCH --account=pi-jjberg
#SBATCH --partition=caslake
#SBATCH --time=25:00:00
#SBATCH --output=posterior_cpu.out
# TO USE V100 specify --constraint=v100
# TO USE RTX600 specify --constraint=rtx6000
#******SBATCH --constraint=v100   # constraint job runs on V100 GPU use
#SBATCH --ntasks=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=2   # set this to the desired number of threads
#SBATCH --mem=32G

module load cuda
module load python

source SFS/bin/activate

python posterior_parallelpopgensim_ac_nfe_lof2.py
