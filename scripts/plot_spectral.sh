#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 20GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.err

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/"
save_path="/proj/rep-learning-robotics/users/x_nonra/data/visualization/spectral"
split="train"

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

SUBJECTS=(1 2 3 4 5 6 7 8 9 10)

for subject_id in "${SUBJECTS[@]}"; do
    echo "subject_id: $subject_id"
    apptainer exec --nv $CONTAINER python src/evaluation/eeg_spectral.py --data_path "$data_path" --save_path "$save_path" --subject_id "$subject_id" --split "$split"
done
