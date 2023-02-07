#!/bin/bash -l

#SBATCH --partition=members
#SBATCH --nodelist=ant9
#SBATCH --job-name=ood
#SBATCH --cpus-per-task=32

echo "Computing job "$SLURM_JOB_ID" on "$(hostname)

srun python3 ood_test.py
