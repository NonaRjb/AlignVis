#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --mem 350GB
#SBATCH --gpus=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=5
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%A_%a_slurm.err
#SBATCH --array=1-10

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
new_label_path=/proj/rep-learning-robotics/users/x_nonra/data/generated_labels
new_label_type=object_high
experiment="new_labels_clip" # "nice_things-eeg-2_insubject"
dataset="things-eeg-2"
seeds=(7 42 191 2025 96723) # Array of seeds  7 42 191 2025 96723

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/


# nice lr = 0.0002 and temperature = 0.04 and batch size = 128 and epoch = 50
# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

nvidia-smi

subject_id=${SLURM_ARRAY_TASK_ID}

echo "subject_id: $subject_id"
echo "Subject list: $subject_list"

# Loop over seeds
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    
    apptainer exec --nv $CONTAINER python src/evaluation/train_new_labels.py --data_path "$data_path" --save_path "$save_path" --new_label_path "$new_label_path" --separate_test \
    --dataset "$dataset" --subject_id "$subject_id" --test_subject "$subject_id" --eeg_enc "eegnet" --epoch 100 --experiment "$experiment" --new_label_type "$new_label_type" \
    -b 128 --n_workers 8 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.04 --scheduler "cosine" --embed_dim 64 &
done
wait