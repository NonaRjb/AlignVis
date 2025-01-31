#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 400GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.err

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/"
save_path="/proj/rep-learning-robotics/users/x_nonra/data/visualization/gradcam"
model_path="/proj/rep-learning-robotics/users/x_nonra/data/eeg_models_trained/nice_things-eeg-2_glocal/"
img_encoder="gLocal_dino-vit-base-p8"
brain_encoder="nice"
subject_id=10
split="test"
seed=42

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ds
# export PYTHONPATH="$PYTHONPATH:$(realpath ./dreamsim)"

apptainer exec --nv $CONTAINER python src/evaluation/eeg_heatmap.py --data_path "$data_path" --save_path "$save_path" --model_path "$model_path" \
 --img_encoder "$img_encoder" --brain_encoder "$brain_encoder" --subject_id 1 2 3 4 5 6 7 8 9 10 --split "$split" --seed 7 42 191 2025 96723
