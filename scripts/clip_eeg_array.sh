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
#SBATCH --array=2-3%2

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data
experiment="mlp_nsd-fmri_insubject" # "nice_things-eeg-2_insubject"
img_enc="OpenCLIP_ViT-B32_laion400m_noalign"
img_enc_noalign="harmonization_levit"
dataset="nsd-fmri"
seeds=(7 42 191 2025 96723) # Array of seeds  7 42 191 2025 96723

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/


# nice lr = 0.0002 and temperature = 0.04 and batch size = 128 and epoch = 50
# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

nvidia-smi

subject_id=$((2 * SLURM_ARRAY_TASK_ID + 1))

echo "subject_id: $subject_id"
echo "Subject list: $subject_list"

# Loop over seeds
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    
    apptainer exec --nv $CONTAINER python src/train_brain_clip.py --data_path "$data_path" --save_path "$save_path" --separate_test \
    --dataset "$dataset" --subject_id "$subject_id" --eeg_enc "brain-mlp" --img_enc "$img_enc" --epoch 50 --experiment "$experiment" --img "embedding" \
    --downstream "retrieval" -b 128 --n_workers 8 --lr 0.0001 --warmup 0 --seed "$seed" --temperature 0.07 --scheduler "cosine" --loss clip-loss &
done
wait