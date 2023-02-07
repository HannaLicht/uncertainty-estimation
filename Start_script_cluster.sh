#!/bin/bash -l

#SBATCH --partition=members
#SBATCH --nodelist=ant11
#SBATCH --job-name=active_learn
#SBATCH --cpus-per-task=128

echo "Computing job "$SLURM_JOB_ID" on "$(hostname)

srun python3 Retrainer.py
