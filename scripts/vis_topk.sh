#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 10GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.err

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/"
save_path="/proj/rep-learning-robotics/users/x_nonra/data/visualization/topk"
model_path="/proj/rep-learning-robotics/users/x_nonra/data/eeg_models_trained/"
img_encoder_aligned="dreamsim_clip_vitb32"
img_encoder_noalign="CLIP_ViT-B32_noalign"
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

apptainer exec --nv $CONTAINER python src/evaluation/visualize_topk_images.py --data_path "$data_path" --save_path "$save_path" --model_path "$model_path" \
 --img_encoder_aligned "$img_encoder_aligned" --img_encoder_noalign "$img_encoder_noalign" --brain_encoder "nice" --subject_id "$subject_id" --split "$split" --seed "$seed"



# python alignvis/src/evaluation/visualize_topk_images.py --data_path "/proj/rep-learning-robotics/users/x_nonra/alignvis/data/" --save_path "/proj/rep-learning-robotics/users/x_nonra/data/visualization/topk" \
# --model_path "/proj/rep-learning-robotics/users/x_nonra/data/eeg_models_trained/" --brain_encoder nice --subject_id 10 --img_encoder_aligned "dreamsim_synclr_vitb16" \
# --img_encoder_noalign "dreamsim_synclr_vitb16_noalign" --split test --seed 42