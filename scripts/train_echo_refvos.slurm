#!/bin/bash
#SBATCH --job-name=samwise-echo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ultrai/UltrAi/moein/SAMWISE}"
cd "${REPO_DIR}"

EXP_NAME="${EXP_NAME:-echo_refvos_train}"
DATA_ROOT="${DATA_ROOT:-data/echo-ref-vos}"
RESUME_CKPT="${RESUME_CKPT:-pretrain/pretrained_model.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"

GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
MASTER_PORT="${MASTER_PORT:-29500}"

# In this codebase, --batch_size is treated as a global batch size and then
# divided by the number of GPUs in main.py.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FRAMES="${NUM_FRAMES:-8}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Experiment: ${EXP_NAME}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  --dataset_file ytvos \
  --ytvos_path "${DATA_ROOT}" \
  --resume "${RESUME_CKPT}" \
  --output_dir "${OUTPUT_ROOT}" \
  --name_exp "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --batch_size "${GLOBAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --num_frames "${NUM_FRAMES}" \
  --HSA \
  --use_cme_head
