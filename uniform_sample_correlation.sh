#!/bin/bash
#SBATCH --job-name=FastSampling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=job_%x.%j.log
#SBATCH --error=job_%x.%j.log
#SBATCH --hint=nomultithread


# echo job info on joblog:
echo "Job ID: ${SLURM_JOB_ID}"
echo "Hostname: $(hostname -s)"
echo "Start date: $(date)"

source ~/.bashrc

mamba activate tenpy

#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1

export N=10000
export L_LIST=20
export BETA_LIST=0.1,0.2,0.3,0.5,0.8,1.0
export N_WORKERS=16
python uniform_sample_correlation.py
