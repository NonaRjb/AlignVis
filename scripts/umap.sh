#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 10GB
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.err

data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/"
save_path="/proj/rep-learning-robotics/users/x_nonra/data/visualization"
model_path_aligned="/proj/rep-learning-robotics/users/x_nonra/data/eeg_models_trained/dreamsim_synclr_vitb16/sub-10/models/nice_things-eeg-2_seed42_49.pth"
model_path_noalign="/proj/rep-learning-robotics/users/x_nonra/data/eeg_models_trained/dreamsim_synclr_vitb16_noalign/sub-10/models/nice_things-eeg-2_seed42_47.pth"
img_encoder_aligned="dreamsim_synclr_vitb16"
img_encoder_noalign="dreamsim_synclr_vitb16_noalign"
subject_id=10
split="test"


cd /proj/rep-learning-robotics/users/x_nonra/alignvis/

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate ds
# export PYTHONPATH="$PYTHONPATH:$(realpath ./dreamsim)"

python src/evaluation/umap_visualization.py --data_path "$data_path" --save_path "$save_path" --model_path_aligned "$model_path_aligned" --model_path_noalign "$model_path_noalign" \
 --img_encoder_aligned "$img_encoder_aligned" --img_encoder_noalign "$img_encoder_noalign" --brain_encoder "nice" --normalize --subject_id "$subject_id" --split "$split"
