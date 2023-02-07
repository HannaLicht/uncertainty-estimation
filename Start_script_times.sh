#!/bin/bash -l

#SBATCH --partition=ci
#SBATCH --nodelist=ant12
#SBATCH --job-name=times
#SBATCH --cpus-per-task=64

echo "Computing job "$SLURM_JOB_ID" on "$(hostname)

srun python3 Ensemble_uncertainty_test.py

