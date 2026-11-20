"""CPU tests: the per-rank sampled replay resolves each declared request's Program DIR by the manifest's lookup of record
(`required_manifest._program_dirs_for`), so a B>=2 mixed workload's nominal-shape request -- which row_pod builds under the bare `build_request/`
(result.json LP/T), not `build_request_LP<lp>_T<t>` -- is found, and a missing or near shape is still refused by name (never a near shape)."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from verity_vllm.pipeline.tp.rank_replay import request_program_dirs


def _prog(root, name, lp, t, ok=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"ok": ok, "LP": lp, "T": t}))
    return str(d)


def _req(rid, lp, max_tokens):
    return {"request_id": rid, "prompt_len": lp, "max_tokens": max_tokens}


def test_nominal_shape_under_bare_build_request_and_named_shape_both_resolve(tmp_path):
    # the Qwen3-30B-A3B TP2 B2 1024/128 mixed layout: r0 = (LP 1024, T 127) under build_request/, r1 = (LP 9, T 1) under build_request_LP9_T1/
    root = tmp_path / "rank0"
    nominal = _prog(root, "build_request", 1024, 127)
    small = _prog(root, "build_request_LP9_T1", 9, 1)
    dirs = request_program_dirs(str(root), [_req("r0", 1024, 128), _req("r1", 9, 2)], 0)
    assert dirs == {"r0": nominal, "r1": small}


def test_named_lp_t_dir_wins_over_the_bare_default(tmp_path):
    root = tmp_path / "rank1"
    _prog(root, "build_request", 1024, 127)
    named = _prog(root, "build_request_LP1024_T127", 1024, 127)
    assert request_program_dirs(str(root), [_req("r0", 1024, 128)], 1) == {"r0": named}


def test_missing_shape_is_refused_by_name_with_the_rank(tmp_path):
    root = tmp_path / "rank0"
    _prog(root, "build_request", 1024, 127)
    with pytest.raises(RuntimeError) as ei:
        request_program_dirs(str(root), [_req("r0", 1024, 128), _req("r1", 100, 6)], 0)
    msg = str(ei.value)
    assert msg.startswith("rank 0:") and "LP100_T5" in msg and "r1" in msg


def test_near_shape_and_failed_build_are_never_taken(tmp_path):
    root = tmp_path / "rank0"
    _prog(root, "build_request", 1023, 127)  # near shape: LP off by one
    _prog(root, "build_request_x", 1024, 127, ok=False)  # exact shape but the Build did not pass
    with pytest.raises(RuntimeError, match=r"rank 0: .*LP1024_T127"):
        request_program_dirs(str(root), [_req("r0", 1024, 128)], 0)


def test_single_program_root_maps_every_request_to_the_root(tmp_path):
    root = tmp_path / "rank0" / "build_request"
    root.mkdir(parents=True)
    (root / "instances.json.gz").write_bytes(b"")
    dirs = request_program_dirs(str(root), [_req("r0", 256, 32), _req("r1", 256, 32)], 0)
    assert dirs == {"r0": str(root), "r1": str(root)}
