#!/usr/bin/env bash
# Submit the self-contained two-node NanoV3 NeMo-RL rollout from this worktree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# The login host resolves this worktree through fs1; compute-node containers
# mount the same shared storage through fsw.
CONTAINER_SCRIPT_DIR="${SCRIPT_DIR/\/lustre\/fs1\//\/lustre\/fsw\/}"

SHARED_ROOT="${SHARED_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/chunweiy}"
IMAGE_ROOT="${IMAGE_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/shuyix}"
CONTAINER="${CONTAINER:-$IMAGE_ROOT/images/trtllm_base-e80b2876e-aarch64.sqsh}"
TRTLLM_BUILD="${TRTLLM_BUILD:-$IMAGE_ROOT/TRTLLM_BUILD}"
MODEL_CACHE="${HF_HOME:-$IMAGE_ROOT/hf_cache}"
DATA_CACHE="${HF_DATASETS_CACHE:-$SHARED_ROOT/hf_cache/datasets}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-2}"
JOB_NAME="${JOB_NAME:-nemorl-nanov3-trtllm}"
MAX_STEPS="${MAX_STEPS:-1}"

[[ -f "$CONTAINER" ]] || { echo "CONTAINER not found: $CONTAINER" >&2; exit 2; }
mkdir -p "$DATA_CACHE"
MOUNTS="$CONTAINER_SCRIPT_DIR:/opt/nemo-rl,$CONTAINER_SCRIPT_DIR:$CONTAINER_SCRIPT_DIR,$TRTLLM_BUILD:$TRTLLM_BUILD,$MODEL_CACHE:$MODEL_CACHE,$DATA_CACHE:$DATA_CACHE"

# Set PROFILE=1 only after selecting a short, calibrated global engine range,
# e.g. TLLM_PROFILE_START_STOP=25,30. CUDA graphs and overlap are untouched.
if [[ "${PROFILE:-0}" == 1 ]]; then
  : "${TLLM_PROFILE_START_STOP:?PROFILE=1 requires TLLM_PROFILE_START_STOP}"
  export NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-trtllm_*generation_worker}"
  export NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-$TLLM_PROFILE_START_STOP}"
  export TLLM_LLMAPI_ENABLE_NVTX=1
else
  unset NRL_NSYS_WORKER_PATTERNS NRL_NSYS_PROFILE_STEP_RANGE TLLM_PROFILE_START_STOP TLLM_LLMAPI_ENABLE_NVTX || true
fi

export COMMAND="source '$CONTAINER_SCRIPT_DIR/node_init_script.sh' && uv run python -u examples/run_grpo.py --config examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n4g-fsdp2-trtllm.yaml grpo.max_num_steps=$MAX_STEPS logger.wandb.mode=${WANDB_MODE:-offline}"
export CONTAINER MOUNTS GPUS_PER_NODE NNODES JOB_NAME
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-30}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-$CONTAINER_SCRIPT_DIR/logs}"

sbatch --nodes="$NNODES" --gres="gpu:$GPUS_PER_NODE" --account="${SLURM_ACCOUNT:-coreai_comparch_trtllm}" --job-name="$JOB_NAME" --partition="${SLURM_PARTITION:-batch}" --time="${SLURM_TIME:-4:00:00}" --exclusive ray.sub
