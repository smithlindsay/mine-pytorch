#!/bin/bash
#SBATCH --job-name=gabor_bias_unbias   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G        
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)

python3 gabor_bias_unbias.py
