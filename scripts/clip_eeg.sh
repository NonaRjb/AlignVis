#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 500GB
#SBATCH --gpus=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=5
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.err

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/eeg_torch_container.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
experiment="nice_things-eeg-2_cross_subject"
img_enc="dreamsim_synclr_vitb16"
dataset="things-eeg-2"
seed=2025
subject_id=7
subject_list=$(seq 1 10 | grep -v "^$subject_id$")
# Convert the newline-separated list into space-separated integers
subject_args=$(echo $subject_list)

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

nvidia-smi

apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
--dataset "$dataset" --subject_id $subject_args --test_subject "$subject_id" --eeg_enc "nice" --img_enc "$img_enc" --epoch 150 --experiment "$experiment" --img "embedding" \
--downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine"
# apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
# --dataset "$dataset" --subject_id "$subject_id" --eeg_enc "nice" --img_enc "${img_enc}_noalign" --epoch 50 --experiment "$experiment" --img "embedding" \
# --downstream "retrieval" -b 128 --n_workers 8 --lr 0.0002 --warmup 0 --seed 42 --temperature 0.04 --scheduler "cosine" &
# wait

# python src/train_brain_clip.py --data_path /proj/rep-learning-robotics/users/x_nonra/alignvis/data --save_path /proj/rep-learning-robotics/users/x_nonra/data/ --dataset things-eeg-2 --subject_id 1 --eeg_enc nice --img_enc dreamsim_clip_vitb32 --epoch 10 --experiment test_nice --img "embedding" --downstream retrieval -b 512 --n_workers 4 --lr 0.0002 --warmup 0 --seed 42 --temperature 0.04 --scheduler cosine --patience 5