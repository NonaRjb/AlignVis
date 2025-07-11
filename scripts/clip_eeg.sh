#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 400GB
#SBATCH --gpus=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=5
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.err
#SBATCH --array=1-1

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
checkpoint="/proj/rep-learning-robotics/users/x_nonra/data/nice_things-eeg-2_cross_subject/dreamsim_ensemble_noalign/sub-10/models/nice_things-eeg-2_seed42_59.pth"
experiment="nice_things-eeg-2_cross_subject"
img_enc="dreamsim_ensemble_noalign"
dataset="things-eeg-2"
seeds=(42) # Array of seeds 7 42 191 2025 96723
subject_id=10
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

    apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --checkpoint "$checkpoint" --separate_test \
    --dataset "$dataset" --subject_id $subject_args --test_subject $subject_id --eeg_enc "nice" --img_enc "$img_enc" --epoch 0 --experiment "$experiment" --img "embedding" \
    --downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine"
    # apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
    # --dataset "$dataset" --subject_id 1 2 3 4 5 6 7 9 10 --test_subject 8 --eeg_enc "nice" --img_enc "$img_enc" --epoch 150 --experiment "$experiment" --img "embedding" \
    # --downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine" &
    # wait
done