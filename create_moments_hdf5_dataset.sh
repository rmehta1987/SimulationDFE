#!/bin/bash

#SBATCH --job-name=momcache
#SBATCH --account=pi-jjberg
#SBATCH --partition=caslake
#SBATCH --time=16:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --mem=18G
#SBATCH --output=create_moments.out



module load python

source SFS/bin/activate

python cache_moments_data.py
