#!/bin/bash

#SBATCH --job-name=cdata
#SBATCH --account=pi-jjberg
#SBATCH --partition=caslake
#SBATCH --time=12:00:00
#SBATCH --mem=18G
#SBATCH --output=create_dataset.out
# TO USE V100 specify --constraint=v100
# TO USE RTX600 specify --constraint=rtx6000
#******SBATCH --constraint=v100   # constraint job runs on V100 GPU use
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=2   # set this to the desired number of threads


module load python

source SFS/bin/activate

python create_sim_dataset.py
