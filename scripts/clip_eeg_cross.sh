#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 800GB
#SBATCH --gpus=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=5
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.err
#SBATCH --array=1-10

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/eeg_torch_container.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
experiment="nice_things-eeg-2_cross_subject" # "nice_things-eeg-2_insubject"
img_enc="gLocal_dino-vit-base-p16"
img_enc_noalign="DINO_ViT-B16_noalign"
dataset="things-eeg-2"
seed=7 # 7, 42, 191, 2025, 96723

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

nvidia-smi

subject_id=${SLURM_ARRAY_TASK_ID}

subject_list=$(seq 1 10 | grep -v "^$subject_id$")
# Convert the newline-separated list into space-separated integers
subject_args=$(echo $subject_list)

echo "Excluding subject_id: $subject_id"
echo "Subject list: $subject_list"

apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
--dataset "$dataset" --subject_id $subject_args --test_subject "$subject_id" --eeg_enc "nice" --img_enc "$img_enc" --epoch 150 --experiment "$experiment" --img "embedding" \
--downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine" &
apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
--dataset "$dataset" --subject_id $subject_args --test_subject "$subject_id" --eeg_enc "nice" --img_enc "$img_enc_noalign" --epoch 150 --experiment "$experiment" --img "embedding" \
--downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine" &
wait
