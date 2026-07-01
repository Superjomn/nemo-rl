"""Focused contract tests for TRT-LLM's inner-Ray nsys integration."""

from pathlib import Path
from unittest.mock import patch

from nemo_rl.distributed.worker_group_utils import get_trtllm_ray_worker_nsight_options


def test_trtllm_inner_worker_nsys_is_disabled_without_profile_environment():
    with (
        patch("nemo_rl.distributed.worker_group_utils.NRL_NSYS_WORKER_PATTERNS", ""),
        patch("nemo_rl.distributed.worker_group_utils.NRL_NSYS_PROFILE_STEP_RANGE", ""),
    ):
        assert get_trtllm_ray_worker_nsight_options("trtllm_generation_worker") is None


def test_trtllm_inner_worker_nsys_matches_only_matching_worker():
    with (
        patch(
            "nemo_rl.distributed.worker_group_utils.NRL_NSYS_WORKER_PATTERNS",
            "trtllm_*generation_worker",
        ),
        patch("nemo_rl.distributed.worker_group_utils.NRL_NSYS_PROFILE_STEP_RANGE", "25,30"),
    ):
        options = get_trtllm_ray_worker_nsight_options("trtllm_generation_worker")
        assert options is not None
        assert options["capture-range"] == "cudaProfilerApi"
        assert options["t"] == "cuda,nvtx"
        assert get_trtllm_ray_worker_nsight_options("policy_worker") is None


def test_sync_and_async_workers_forward_without_overriding_user_kwargs():
    root = Path(__file__).parents[4]
    for relative in (
        "nemo_rl/models/generation/trtllm/trtllm_worker.py",
        "nemo_rl/models/generation/trtllm/trtllm_worker_async.py",
    ):
        source = (root / relative).read_text()
        assert "get_trtllm_ray_worker_nsight_options" in source
        # setdefault is the documented precedence: user advanced kwargs win.
        assert 'llm_kwargs.setdefault("ray_worker_nsight_options", nsight_options)' in source


def test_rank0_executor_patch_is_idempotent_and_has_all_required_anchors():
    root = Path(__file__).parents[4]
    source = (root / "node_init_script.sh").read_text()
    assert "NEMO_RL_RANK0_NSYS_GATE" in source
    assert "rank == 0" in source
    assert "src.count(needle) != 1" in source
    assert 'pop("nsight", None)' in source
