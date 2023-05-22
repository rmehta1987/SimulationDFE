#!/bin/bash

#SBATCH --job-name=mispost
#SBATCH --account=pi-jjberg
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=posterior_mis.out
# TO USE V100 specify --constraint=v100
# TO USE RTX600 specify --constraint=rtx6000
#******SBATCH --constraint=v100   # constraint job runs on V100 GPU use
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=2   # set this to the desired number of threads


module load cuda
module load python

source SFS/bin/activate

python posterior_parallelpopgensim_ac_nfe_mis_classifier.py
