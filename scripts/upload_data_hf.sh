#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --mem 50GB
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_upload_data.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_upload_data.err

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate alignvis

#  python /proj/rep-learning-robotics/users/x_nonra/alignvis/scripts/upload_data_hf.py
python /proj/rep-learning-robotics/users/x_nonra/alignvis/scripts/create_wds_shards.py
