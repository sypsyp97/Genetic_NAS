#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --job-name=genetic_nas
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH -o /home/hpc/iwb3/iwb3005h/output/slurm-%j.out
#SBATCH -e /home/hpc/iwb3/iwb3005h/output/slurm-%j.err
#
# do no export environment variables
#SBATCH --export=NONE
#
# do not export environment variables
unset SLURM_EXPORT_ENV
#
# load required modules
module load python/3.8-anaconda
#
# anaconda environment
source activate tf
#
# configure paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#
# run
python tf_flower_test.py ${WORK}/
