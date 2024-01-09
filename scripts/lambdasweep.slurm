#!/bin/bash
#SBATCH --job-name=lambdasweep   # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=100G         # memory per cpu-core (4G is default)
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-6              # job array with index values 0, 1, 2, 3, 4

LAMS=(0.0005 0.001 0.005 0.01 0.05 0.1 0.5)

python lambdasweep.py ${LAMS[$SLURM_ARRAY_TASK_ID]}
