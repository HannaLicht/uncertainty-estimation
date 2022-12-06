#!/bin/bash -l

#SBATCH --time=23:00:00
#SBATCH --partition=members
#SBATCH --nodelist=ant9
#SBATCH --job-name=active_learning_uncertainty
#SBATCH --cpus-per-task=32

echo "Computing job "$SLURM_JOB_ID" on "$(hostname)

srun python3 Evaluation/Retrainer.py
