#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "${REPO_DIR}"

YTVOS_PATH="${YTVOS_PATH:-data/echo-ref-vos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
TRAIN_EXP_NAME="${TRAIN_EXP_NAME:-echo_refvos_train}"
EPOCH="${EPOCH:-3}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_ROOT}/${TRAIN_EXP_NAME}/checkpoint$(printf '%04d' "${EPOCH}").pth}"
P2FLOW_ROOT="${P2FLOW_ROOT:-/home/ultrai/UltrAi/moein/P2Flow}"

TEST_GPU="${TEST_GPU:-0}"
EXTERNAL_GPU="${EXTERNAL_GPU:-1}"

TEST_EXP_NAME="${TEST_EXP_NAME:-${TRAIN_EXP_NAME}_test_epoch$(printf '%02d' "${EPOCH}")}"
EXTERNAL_EXP_NAME="${EXTERNAL_EXP_NAME:-${TRAIN_EXP_NAME}_external_epoch$(printf '%02d' "${EPOCH}")}"

TEST_SKIP_DATASETS="${TEST_SKIP_DATASETS:-}"
EXTERNAL_SKIP_DATASETS="${EXTERNAL_SKIP_DATASETS:-EchoNet-Dynamic}"

COMMON_INFER_ARGS=(
  --dataset_file ytvos
  --ytvos_path "${YTVOS_PATH}"
  --resume "${CHECKPOINT}"
  --output_dir "${OUTPUT_ROOT}"
  --no_distributed
  --HSA
  --use_cme_head
)

COMMON_SCORE_ARGS=(
  --ytvos-path "${YTVOS_PATH}"
  --p2flow-root "${P2FLOW_ROOT}"
)

echo "Checkpoint: ${CHECKPOINT}"
echo "Test inference GPU: ${TEST_GPU}"
echo "External inference GPU: ${EXTERNAL_GPU}"
echo "Test experiment: ${TEST_EXP_NAME}"
echo "External experiment: ${EXTERNAL_EXP_NAME}"

echo
echo "[1/4] Running test inference"
CUDA_VISIBLE_DEVICES="${TEST_GPU}" uv run python inference_echo.py \
  "${COMMON_INFER_ARGS[@]}" \
  --name_exp "${TEST_EXP_NAME}" \
  --split test \
  --skip_datasets "${TEST_SKIP_DATASETS}"

echo
echo "[2/4] Scoring test predictions"
uv run python scripts/eval_echo_epoch_p2flow_metrics.py \
  --epoch-dir "${OUTPUT_ROOT}/${TEST_EXP_NAME}" \
  --split test \
  --skip_datasets "${TEST_SKIP_DATASETS}" \
  "${COMMON_SCORE_ARGS[@]}"

echo
echo "[3/4] Running external inference"
CUDA_VISIBLE_DEVICES="${EXTERNAL_GPU}" uv run python inference_echo.py \
  "${COMMON_INFER_ARGS[@]}" \
  --name_exp "${EXTERNAL_EXP_NAME}" \
  --split external \
  --skip_datasets "${EXTERNAL_SKIP_DATASETS}"

echo
echo "[4/4] Scoring external predictions"
uv run python scripts/eval_echo_epoch_p2flow_metrics.py \
  --epoch-dir "${OUTPUT_ROOT}/${EXTERNAL_EXP_NAME}" \
  --split external \
  --skip_datasets "${EXTERNAL_SKIP_DATASETS}" \
  "${COMMON_SCORE_ARGS[@]}"

echo
echo "Done."
echo "Test outputs: ${OUTPUT_ROOT}/${TEST_EXP_NAME}/eval_echo/test"
echo "External outputs: ${OUTPUT_ROOT}/${EXTERNAL_EXP_NAME}/eval_echo/external"
