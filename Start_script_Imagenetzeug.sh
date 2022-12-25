#!/bin/bash -l

#SBATCH --time=23:59:00
#SBATCH --partition=members
#SBATCH --job-name=efficientnet
#SBATCH --cpus-per-task=64

echo "Computing job "$SLURM_JOB_ID" on "$(hostname)

srun python3 MCdropout_uncertainty_test.py
