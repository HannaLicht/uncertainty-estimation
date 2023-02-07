#!/bin/bash -l

#SBATCH --partition=ci
#SBATCH --nodelist=ant12
#SBATCH --job-name=metrics
#SBATCH --cpus-per-task=64

echo "Computing job "$SLURM_JOB_ID" on "$(hostname)

srun python3 evaluation_metrics.py
