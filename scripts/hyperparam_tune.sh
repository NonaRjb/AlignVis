#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 800GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.err
#SBATCH --array=0-11 # Total combinations of temperatures and batch sizes

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
experiment="hyperparam_tune"
img_enc="dreamsim_clip_vitb32"
dataset="things-eeg-2"

temperatures=(0.04 0.07 0.1)
batch_sizes=(128 256 512 1024)

temp_idx=$((SLURM_ARRAY_TASK_ID / ${#batch_sizes[@]}))
batch_idx=$((SLURM_ARRAY_TASK_ID % ${#batch_sizes[@]}))

temperature=${temperatures[$temp_idx]}
batch_size=${batch_sizes[$batch_idx]}

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo "TEMPERATURE: ${temperature}"
echo "BATCH SIZE: ${batch_size}"
echo ""

nvidia-smi

apptainer exec --nv $CONTAINER python src/train_brain_clip.py \
  --data_path "$data_path" \
  --save_path "$save_path" \
  --separate_test \
  --dataset "$dataset" \
  --subject_id 1 \
  --eeg_enc "eegconformer" \
  --img_enc "$img_enc" \
  --epoch 200 \
  --experiment "$experiment" \
  --img "embedding" \
  --downstream "retrieval" \
  -b "$batch_size" \
  --n_workers 8 \
  --lr 0.0002 \
  --warmup 0 \
  --seed 42 \
  --temperature "$temperature" \
  --scheduler "cosine"
