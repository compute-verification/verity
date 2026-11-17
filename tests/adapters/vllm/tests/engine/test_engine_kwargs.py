"""`engine_profile.engine_kwargs`: the `LLM(...)` keywords of an engine the adapter builds, flag by flag, and the refusal of a
replay engine.  There is none: the replay evaluates a Build (`check/replay`).  CPU-only: no engine is built."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.engine import engine_profile as prof

SEED = 7

#: each declared flag, a value that sets it, and the keys that value adds to the stock keywords
FLAGS: dict[str, tuple[Any, dict[str, Any]]] = {
    "enforce_eager": (True, {"enforce_eager": True}),
    "custom_ops": ("all", {"compilation_config": {"custom_ops": ["all"]}}),
    "max_num_batched_tokens": (2048, {"max_num_batched_tokens": 2048}),
    "tensor_parallel_size": (2, {"tensor_parallel_size": 2}),
    "quantization": ("fp8", {"quantization": "fp8"}),
}

#: the values that leave a flag unset: its key stays out, so a profile id made without the flag stays what it was
UNSET: dict[str, tuple[Any, ...]] = {
    "enforce_eager": (False,),
    "custom_ops": (None, ""),
    "max_num_batched_tokens": (None,),
    "tensor_parallel_size": (None, 1),
    "quantization": (None, ""),
}


def _stock(snapshot: Path) -> dict[str, Any]:
    return {"model": str(snapshot), "tokenizer": str(snapshot), "dtype": "bfloat16", "seed": SEED,
            "max_model_len": prof.MAX_MODEL_LEN, "gpu_memory_utilization": prof.GPU_MEMORY_UTILIZATION, "trust_remote_code": False}


def test_with_no_flag_the_keywords_are_the_stock_ones(tmp_path: Path) -> None:
    assert prof.engine_kwargs({"local_path": str(tmp_path)}, replay=False, seed=SEED) == _stock(tmp_path)


@pytest.mark.parametrize("flag", sorted(FLAGS))
def test_a_declared_flag_adds_its_own_keys_and_no_other(flag: str, tmp_path: Path) -> None:
    cp = {"local_path": str(tmp_path)}
    value, adds = FLAGS[flag]
    assert prof.engine_kwargs(cp, replay=False, seed=SEED, **{flag: value}) == {**_stock(tmp_path), **adds}
    for unset in UNSET[flag]:
        assert prof.engine_kwargs(cp, replay=False, seed=SEED, **{flag: unset}) == _stock(tmp_path), unset


def test_every_flag_at_once_adds_each_flags_keys(tmp_path: Path) -> None:
    got = prof.engine_kwargs({"local_path": str(tmp_path)}, replay=False, seed=SEED, **{flag: value for flag, (value, _) in FLAGS.items()})
    assert got == {**_stock(tmp_path), **{key: value for _, adds in FLAGS.values() for key, value in adds.items()}}


@pytest.mark.parametrize("flags", [{}, {flag: value for flag, (value, _) in FLAGS.items()}], ids=["stock", "every-flag"])
def test_a_replay_engine_is_refused_and_the_refusal_names_where_the_replay_is(flags: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="there is no replay engine") as refusal:
        prof.engine_kwargs({"local_path": str(tmp_path)}, replay=True, seed=SEED, **flags)
    assert "check/replay" in str(refusal.value) and "replay=False" in str(refusal.value)
    assert importlib.util.find_spec("verity_vllm.check.replay") is not None
