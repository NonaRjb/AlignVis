#!/usr/bin/env bash
#SBATCH -A berzelius-2025-35
#SBATCH --mem 10GB
#SBATCH --gpus=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=5
#SBATCH -t 10:00:00
#SBATCH --mail-type FAIL
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/alignvis/logs/%J_slurm.err

CONTAINER=/proj/rep-learning-robotics/users/x_nonra/containers/alignvis.sif
save_path=/proj/rep-learning-robotics/users/x_nonra/data/visualization/corr
data_path=/proj/rep-learning-robotics/users/x_nonra/data/encoding_model_ridge_preprocessed
img_enc_1="dreamsim_clip_vitb32"
img_enc_2="CLIP_ViT-B32_noalign"

cd /proj/rep-learning-robotics/users/x_nonra/alignvis/


# nice lr = 0.0002 and temperature = 0.04 and batch size = 128 and epoch = 50
# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

nvidia-smi

subject_ids=(1 2 3 4 5 6 7 8 9 10)

echo "subject_id: $subject_id"
echo "Subject list: $subject_list"

# Loop over subject_ids
for sid in "${subject_ids[@]}"; do
    apptainer exec --nv $CONTAINER python src/evaluation/visualize_corr.py --data_path "$data_path" --save_path "$save_path" \
        --subject_id "$sid" --dnn1 "$img_enc_1" --dnn2 "$img_enc_2"
done
