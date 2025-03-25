#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --mem 50GB
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
data_path=/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2
experiment="encoding_model_ridge_preprocessed" # "nice_things-eeg-2_insubject"
img_enc="dreamsim_clip_vitb32"
dataset="things-eeg-preprocessed"
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
apptainer exec --nv $CONTAINER python src/evaluation/train_encoding_model_reg.py --project_dir "$data_path" --save_dir "$save_path" \
    --sub "$subject_id" --dnn "$img_enc" --n_iter 100 --experiment "$experiment" --n_img_cond 16540 --n_eeg_rep 4 --seed 42 \
    --dataset "$dataset" --t_start -0.1 --t_end 0.6
