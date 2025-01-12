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
#SBATCH --array=1-2

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/eeg_torch_container.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
checkpoint="/proj/rep-learning-robotics/users/x_nonra/data/nice_things-meg_insubject/DINO_ViT-B8_noalign/sub-03/models/nice_things-meg_seed191_35.pth"
experiment="nice_things-eeg-2_cross_subject"
img_enc="dreamsim_synclr_vitb16"
dataset="things-eeg-2"
seeds=(2025 96723) # Array of seeds 7 42 191 2025 96723
subject_id=3
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

# Loop over seeds
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"

    apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
    --dataset "$dataset" --subject_id 1 2 4 5 6 7 8 9 10 --test_subject 3 --eeg_enc "nice" --img_enc "$img_enc" --epoch 150 --experiment "$experiment" --img "embedding" \
    --downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine" &
    apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
    --dataset "$dataset" --subject_id 1 2 3 4 5 6 7 9 10 --test_subject 8 --eeg_enc "nice" --img_enc "$img_enc" --epoch 150 --experiment "$experiment" --img "embedding" \
    --downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine" &
    wait
done