"""Shared test helpers (importable as ``_helpers`` because conftest puts tests/ on sys.path)."""

from __future__ import annotations

from pathlib import Path

TINY_MANIFEST = {
    "schema": "verity-vllm/profile/v1",
    "model": {"repo": "toy/tiny", "revision": "0", "weights_sha256": {}},
    "vllm": {"version": "0.0"},
    "dtype": "bfloat16",
}


def make_tiny_run(run_dir: Path, *, on_duplicate: str = "error"):
    """A two-step capture (prefill of 3 positions, one decode) with weights and a boundary."""
    import numpy as np

    from verity_vllm.capture import Boundary, RunWriter, StepRecord
    from verity_vllm.identity import value_id

    run = RunWriter(run_dir, TINY_MANIFEST)
    pid = run.profile_id
    ru = run.ru("req-0", 0, phase="prefill", positions=[0, 1, 2], on_duplicate=on_duplicate)
    for t in range(3):
        emb = str(value_id(pid, "req-0", t, "prefill", "embed", "embed_tokens", 0))
        ru.add(emb, "bfloat16", [4], np.full(4, 0x3F80 + t, np.uint16))
        res = str(value_id(pid, "req-0", t, "prefill", 0, "layers.0.residual", 0))
        ru.add(res, "bfloat16", [4], np.full(4, 0x4000 + t, np.uint16))
    sel = str(value_id(pid, "req-0", 2, "prefill", "sampler", "sampler.select", 0))
    ru.add(sel, "int32", [], np.array(7, np.int32))
    ru.finish()
    ru = run.ru("req-0", 1, phase="decode", positions=[3], on_duplicate=on_duplicate)
    emb = str(value_id(pid, "req-0", 3, "decode", "embed", "embed_tokens", 0))
    ru.add(emb, "bfloat16", [4], np.full(4, 0x3F83, np.uint16))
    sel = str(value_id(pid, "req-0", 3, "decode", "sampler", "sampler.select", 0))
    ru.add(sel, "int32", [], np.array(9, np.int32))
    ru.finish()
    boundary = Boundary.build(
        "req-0",
        [1, 2, 3],
        [7, 9],
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 3, "seed": None},
        seed=None,
        finish_reason="stop",
        steps=[StepRecord(0, "prefill", (0, 1, 2)), StepRecord(1, "decode", (3,))],
    )
    run.write_boundary(boundary)
    w = run.weights(store_values=True)
    w.add("embed_tokens.weight", "bfloat16", [16, 4], np.arange(64, dtype=np.uint16))
    w.add("lm_head.weight", "bfloat16", [4, 16], np.arange(64, dtype=np.uint16) + 1)
    w.finish()
    run.finish()
    return run
