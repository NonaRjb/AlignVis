#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 800GB
#SBATCH --gpus=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=5
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.err
#SBATCH --array=1-10

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
experiment="nice_things-eeg-2_cross_subject" # "nice_things-eeg-2_insubject"
img_enc="dreamsim_ensemble_noalign"
img_enc_noalign="dreamsim_dino_vitb16"
dataset="things-eeg-2"
seeds=(7 42 191 2025 96723) # Array of seeds

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

# Loop over seeds
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"

    apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
    --dataset "$dataset" --subject_id $subject_args --test_subject "$subject_id" --eeg_enc "nice" --img_enc "$img_enc" --epoch 150 --experiment "$experiment" --img "embedding" \
    --downstream "retrieval" -b 512 --n_workers 10 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine" &
done
wait
