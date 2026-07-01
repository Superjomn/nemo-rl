#!/usr/bin/env bash
# Environment and narrowly-scoped runtime patches for the NanoV3 TRT-LLM job.
# This file is sourced by ray.sub inside every container before `ray start`.

set -euo pipefail

export RAY_DEDUP_LOGS=0
export TRTLLM_UCX_INTERFACE="${TRTLLM_UCX_INTERFACE:-eth0}"
export TLLM_LOG_LEVEL="${TLLM_LOG_LEVEL:-info}"
export NEMO_RL_PY_EXECUTABLES_TRTLLM="${NEMO_RL_PY_EXECUTABLES_TRTLLM:-/usr/bin/python3}"
# TRT-LLM itself is installed in the base image's system interpreter, while
# NeMo-RL dependencies (including transformers) live in this CPython-3.12
# venv.  Make the latter visible to the system-Python Ray actors without
# replacing the interpreter that can import tensorrt_llm.
export PYTHONPATH="/opt/nemo_rl_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export LLM_MODELS_ROOT="${LLM_MODELS_ROOT:-/lustre/fsw/coreai_comparch_trtllm/common}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/shuyix/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PWD}/.cache/hf/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${PWD}/.cache/hf/modules}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(<"$HF_HOME/token")"
fi
# Do not require credentials for a reproducible smoke run.
export WANDB_MODE="${WANDB_MODE:-offline}"

# The inner TRT-LLM Ray actors deliberately use the image's system Python,
# which must have Ray as well as the preinstalled TRT-LLM wheel.
if ! /usr/bin/python3 -c 'import ray' >/dev/null 2>&1; then
  /usr/bin/python3 -m pip install --quiet --no-input "ray[default]==2.54.0"
fi

# TRT-LLM's Python wheel lives in the base-image interpreter.  Ray's nsys
# plugin must retain that interpreter rather than replacing it with `python`.
if [[ -n "${NRL_NSYS_WORKER_PATTERNS:-}" || -n "${NRL_NSYS_PROFILE_STEP_RANGE:-}" ]]; then
  [[ -n "${NRL_NSYS_WORKER_PATTERNS:-}" && -n "${NRL_NSYS_PROFILE_STEP_RANGE:-}" ]] || {
    echo "ERROR: set both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE" >&2
    return 1
  }
  export TLLM_LLMAPI_ENABLE_NVTX=1
  export TLLM_PROFILE_START_STOP="${TLLM_PROFILE_START_STOP:?set a calibrated TRT-LLM iteration range}"
  command -v nsys >/dev/null || { echo "ERROR: nsys not found" >&2; return 1; }
  nsys profile --help | grep -q 'repeat-shutdown' || {
    echo "ERROR: installed nsys lacks capture-range-end=repeat-shutdown" >&2
    return 1
  }
fi

/usr/bin/python3 - <<'PY'
"""Fail-loudly, idempotent patches for the installed Ray/TRT-LLM wheels."""
import glob
import os
import sys

profiling = bool(os.environ.get("NRL_NSYS_WORKER_PATTERNS"))

def patch(path, old, new, label):
    src = open(path).read()
    if new in src:
        print(f"[{label}] already patched: {path}")
        return
    if old not in src:
        raise RuntimeError(f"[{label}] expected anchor missing in {path}; wheel version drift")
    if src.count(old) != 1:
        raise RuntimeError(f"[{label}] expected one anchor in {path}, found {src.count(old)}")
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"[{label}] patched: {path}")

if profiling:
    ray_paths = glob.glob("/opt/nemo_rl_venv/lib*/python*/site-packages/ray/_private/runtime_env/nsight.py")
    ray_paths += glob.glob("/usr/local/lib/python*/dist-packages/ray/_private/runtime_env/nsight.py")
    if not ray_paths:
        raise RuntimeError("[ray-nsys] no installed ray nsight.py found")
    for ray_path in ray_paths:
        patch(
            ray_path,
            'context.py_executable = " ".join(self.nsight_cmd) + " python"',
            'context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"',
            "ray-nsys",
        )

    import importlib.util
    spec = importlib.util.find_spec("tensorrt_llm.executor.ray_executor")
    if spec is None or not spec.origin:
        raise RuntimeError("[trtllm-rank0-nsys] ray_executor not found")
    path = spec.origin
    src = open(path).read()
    marker = "# NEMO_RL_RANK0_NSYS_GATE"
    if marker not in src:
        global_attach = 'runtime_env["nsight"] = ray_worker_nsight_options'
        worker_open = "worker = RayWorkerWrapper.options("
        runtime_kwarg = "runtime_env=runtime_env,"
        for name, needle in (("global nsight attach", global_attach), ("worker construction", worker_open), ("runtime_env kwarg", runtime_kwarg)):
            if src.count(needle) != 1:
                raise RuntimeError(f"[trtllm-rank0-nsys] expected one {name} anchor, found {src.count(needle)}")
        src = src.replace(global_attach, "pass  # NEMO_RL_RANK0_NSYS_GATE: attached per rank")
        block = '''# NEMO_RL_RANK0_NSYS_GATE: trace only the broadcast source (rank 0).\n            import copy as _nemo_rl_copy\n            _nemo_rl_rank_env = _nemo_rl_copy.deepcopy(runtime_env)\n            if ray_worker_nsight_options and rank == 0:\n                _nemo_rl_rank_env["nsight"] = ray_worker_nsight_options\n            else:\n                _nemo_rl_rank_env.pop("nsight", None)\n            '''
        src = src.replace(worker_open, block + worker_open)
        src = src.replace(runtime_kwarg, "runtime_env=_nemo_rl_rank_env,  # NEMO_RL_RANK0_NSYS_GATE")
        with open(path, "w") as f:
            f.write(src)
        print(f"[trtllm-rank0-nsys] patched: {path}")
    else:
        print(f"[trtllm-rank0-nsys] already patched: {path}")
PY
