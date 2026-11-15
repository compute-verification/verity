"""`verity_vllm.check.replay.stoch_recompute`'s inputs and evaluators, apart from a capture: the facts a run's header declares for each
request (`_request_facts`) and those the workload manifest declares (`_manifest_facts`), the CPUs the row evaluations may use
(`_cpu_quota`), and the two evaluators of a sampled row (`_eval_job`, the lane walk of record, and `_eval_job_rows`, its vectorised
twin), which give one token and one masked row.  Torch-free."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity_vllm.check.replay import stoch_recompute as SR
from verity_vllm.records.workload import SCHEMA


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def test_a_requests_facts_are_its_own_then_its_per_request_entry_then_the_runs_sampling():
    header = SimpleNamespace(workload={
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": None,
                     "per_request": [{"request_id": "b", "temperature": 0.5, "top_k": 40}, {"request_id": "c", "top_p": None}]},
        "requests": [{"request_id": "a", "seed": "7"},
                     {"request_id": "b", "seed": 9, "top_p": 0.25, "temperature": None},
                     {"request_id": "c", "top_k": -1}],
        "children": [{"request_id": "a.1", "parent": "a", "seed": 8}, {"request_id": "b.1", "parent": "b"},
                     {"request_id": "x.1", "parent": "x", "seed": 3}]})
    assert SR._request_facts(header) == {
        "a": {"seed": 7, "temperature": 0.8, "top_p": 0.95, "top_k": 0},
        "b": {"seed": 9, "temperature": 0.5, "top_p": 0.25, "top_k": 40},
        "c": {"seed": None, "temperature": 0.8, "top_p": 0.95, "top_k": -1},
        "a.1": {"seed": 8, "temperature": 0.8, "top_p": 0.95, "top_k": 0},           # a child is its parent's facts under its own seed
        "b.1": {"seed": None, "temperature": 0.5, "top_p": 0.25, "top_k": 40},
        "x.1": {"seed": 3}}                                                          # a parent the header does not record gives none


def test_a_header_that_states_no_sampling_declares_greedy_and_one_without_a_workload_declares_nothing():
    header = SimpleNamespace(workload={"requests": [{"request_id": "a", "seed": 1}]})
    assert SR._request_facts(header) == {"a": {"seed": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 0}}
    assert SR._request_facts(SimpleNamespace(workload=None)) == {}
    with pytest.raises(TypeError):                                                   # a temperature stated as null is no number
        SR._request_facts(SimpleNamespace(workload={"sampling": {"temperature": None}, "requests": [{"request_id": "a"}]}))


def _manifest(tmp_path: Path, requests: list[dict], sampling: dict | None) -> Path:
    doc = {"schema": SCHEMA, "model": "m", "engine": {}, "requests": requests,
           "variants": {"stoch-t0.8-p0.95": {"temperature": 0.8, "top_p": 0.95}, "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
                        "stoch-k": {"temperature": 0.7, "top_p": 0.9, "top_k": 50}}}
    p = tmp_path / "wl.json"
    p.write_text(json.dumps(doc if sampling is None else doc | {"sampling": sampling}))
    return p


def test_the_manifests_facts_are_each_requests_variant_and_seed_else_the_workloads(tmp_path):
    wl = _manifest(tmp_path, [{"request_id": "a", "max_tokens": 4}, {"request_id": "b", "max_tokens": 4, "seed": 5},
                              {"request_id": "c", "max_tokens": 4, "variant": "stoch-k"}],
                   {"variant": "stoch-t0.8-p0.95", "seed": 11})
    assert SR._manifest_facts(wl) == {"a": {"seed": 11, "temperature": 0.8, "top_p": 0.95, "top_k": 0},
                                      "b": {"seed": 5, "temperature": 0.8, "top_p": 0.95, "top_k": 0},
                                      "c": {"seed": 11, "temperature": 0.7, "top_p": 0.9, "top_k": 50}}
    assert SR._manifest_facts(_manifest(tmp_path, [{"request_id": "a", "max_tokens": 4}], None)) == {
        "a": {"seed": None, "temperature": 0.0, "top_p": 1.0, "top_k": 0}}              # no sampling declaration: the greedy variant


def test_a_request_that_pins_a_variant_the_manifest_does_not_hold_is_not_read(tmp_path):
    """No record refuses it by name: the variant's parameters are read off None."""
    wl = _manifest(tmp_path, [{"request_id": "a", "max_tokens": 4}, {"request_id": "b", "max_tokens": 4, "variant": "stoch-t9"}],
                   {"variant": "greedy", "seed": None})
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
        SR._manifest_facts(wl)


def test_the_cpu_quota_is_the_cgroups_where_it_states_one_and_the_machines_count_otherwise(tmp_path, monkeypatch):
    """A cgroup file that is not `<quota> <period>` in integers, or cannot be read, states no quota."""
    f = tmp_path / "cpu.max"
    monkeypatch.setattr(SR, "CGROUP_CPU_MAX", str(f))
    machine = os.cpu_count() or 1
    assert SR._cpu_quota() == machine                                                # no cgroup v2 CPU controller
    f.write_text("max 100000\n")
    assert SR._cpu_quota() == machine
    f.write_text("250000 100000\n")
    assert SR._cpu_quota() == 2
    f.write_text("50000 100000\n")
    assert SR._cpu_quota() == 1                                                      # half a CPU is one worker
    for text in ("250000\n", "many 100000\n", "250000 x\n", "1 2 3\n", ""):
        f.write_text(text)
        assert SR._cpu_quota() == machine
    f.unlink()
    f.mkdir()                                                                        # there, and not a file that reads
    assert SR._cpu_quota() == machine


def _row(V: int, top_p: float, temperature: float, seed: int, pos: int) -> list[np.ndarray]:
    """One sampled row's operand words in `recompute_run`'s layout: [bf16 logits row, top_p, seed, pos, temperature]."""
    logits = (np.sin(np.arange(V, dtype=np.float32) * (seed % 7 + 1)) * 3.0).astype(np.float32)
    bf = (logits.view(np.uint32) >> 16).astype(np.uint64)
    return [bf, np.array([_f32(top_p)], dtype=np.uint64), np.array([seed], dtype=np.uint64), np.array([pos], dtype=np.uint64),
            np.array([_f32(temperature)], dtype=np.uint64)]


def test_the_lane_walk_and_its_vectorised_twin_give_one_token_and_one_masked_row():
    jobs = [(64, 4, _row(64, 0.95, 0.8, 1234567, 17)), (64, 2, _row(64, 0.5, 1.0, 99, 3)), (32, 1, _row(32, 1.0, 0.7, 5, 0))]
    for V, S, words in jobs:
        tok, masked, _ = SR._eval_job((V, S, words))
        tok_rows, masked_rows, _ = SR._eval_job_rows((V, S, words))
        assert 0 <= tok < V and tok_rows == tok
        assert masked.dtype == masked_rows.dtype == np.uint32 and masked.shape == (V,) and (masked == masked_rows).all()
    by_lane, by_rows = SR._evaluate_all(jobs, 1, "lane"), SR._evaluate_all(jobs, None, "rows")
    assert [t for t, _, _ in by_lane] == [t for t, _, _ in by_rows] == [SR._eval_job(j)[0] for j in jobs]
    assert SR._evaluate_all([], None, "lane") == []
    assert set(SR.EVALUATORS) == set(SR.EVALUATOR_DOC) == {"lane", "rows"}
