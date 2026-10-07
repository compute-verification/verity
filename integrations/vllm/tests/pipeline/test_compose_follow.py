"""The workload Program composed while the derives run (`global-program --follow`, `components.follow_components`) is the one composed
after them: each component is loaded as its derive finishes, the checks and the composition keep manifest order, a failed derive or
the Build's stop marker composes nothing, and a component still unreadable when the derives are over is left to GP-01's own pass."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch

from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity_vllm.pipeline import components as C
from verity_vllm.pipeline import global_program as G
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.registry import b1

SHAPES = ((3, 2), (5, 1))


def _request_dir(d: Path, LP: int, STEPS: int) -> None:
    """The files of a derive_step request dir that GP-01 reads, for the greedy stand-in of shape (LP, STEPS)."""
    cfg = dict(b1.B0_CONFIG, NL=1, V=64, MAX_POS=16)
    with torch.device("meta"):
        m = ex.GreedyServe(cfg, STEPS)
    rep = derive_report(m, (torch.empty(LP, dtype=torch.int32, device="meta"),), mode="profile", profile="b1-eager-v3", config=cfg,
                        name="GreedyServe")
    assert rep.ok, rep.unsupported
    d.mkdir(parents=True)
    desc = encode_program(rep.program)
    with gzip.open(d / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(desc))
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump({"schema": "cb-a/instance-sequence/v1", "params": [[n, repr(t)] for n, t in rep.flat_definition.signature()[0]], "rows": []}, f)
    json.dump({"structural_inputs": {"LP": LP, "T": STEPS, "sampling": {"kind": "greedy"}}}, open(d / "construction_manifest.json", "w"))
    json.dump({"ok": True, "program": {"digest": program_digest(desc)}, "sampling": {"kind": "greedy"},
               "applicability": {"execution": "bi-eager", "numerics_profile": "stand-in", "sampler": {"kind": "greedy"},
                                 "engine": {"enforce_eager": True, "enable_prefix_caching": False}}}, open(d / "result.json", "w"))


@pytest.fixture(scope="module")
def build_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("row")
    dirs = []
    for lp, t in SHAPES:
        _request_dir(root / f"build_request_LP{lp}_T{t}", lp, t)
        dirs.append(str(root / f"build_request_LP{lp}_T{t}"))
    wl = {"requests": [{"request_id": f"q{i}", "prompt_len": lp, "max_tokens": t + 1, "arrive_step": 0} for i, (lp, t) in enumerate(SHAPES)],
          "sweep": {"row_id": "stand-in__b2__bi-eager"}}
    (root / "workload.json").write_text(json.dumps(wl))
    return root, dirs


def _files(out: Path) -> dict:
    wp = json.loads((out / "workload_program.json").read_text())
    for k in ("cost", "layout", "inputs"):
        wp.pop(k, None)
    return {"descriptor": gzip.open(out / "descriptor.json.gz").read(), "workload_program": wp,
            "address_map": (out / "address_map.json").read_bytes(), "chronology": (out / "chronology.json").read_bytes()}


def test_the_followed_compose_is_the_compose(build_dirs, tmp_path):
    root, dirs = build_dirs
    plain = G.build(str(root / "workload.json"), dirs, str(tmp_path / "plain"))
    canon: dict = {}
    early = C.follow_components(list(reversed(dirs)), set(SHAPES), canon)
    assert set(early) == set(dirs) and all("comp" in e for e in early.values())
    followed = G.build(str(root / "workload.json"), dirs, str(tmp_path / "followed"), early=early, canon=canon)
    assert plain["ok"] and followed["ok"] and plain["workload_digest"] == followed["workload_digest"]
    assert _files(tmp_path / "plain") == _files(tmp_path / "followed")


def test_a_failed_derive_or_the_stop_marker_composes_nothing(build_dirs, tmp_path):
    _root, dirs = build_dirs
    bad = tmp_path / "build_request_LP7_T1"
    bad.mkdir()
    (bad / "result.json").write_text(json.dumps({"ok": False}))
    assert C.follow_components([*dirs, str(bad)], set(SHAPES), {}) is None
    stop = tmp_path / "derives_failed"
    stop.write_text("")
    assert C.follow_components([*dirs, str(tmp_path / "never")], set(SHAPES), {}, stop=str(stop)) is None


def test_once_the_derives_are_over_an_unreadable_dir_is_left_to_the_compose(build_dirs, tmp_path):
    _root, dirs = build_dirs
    done = tmp_path / "derives_done"
    done.write_text("")
    early = C.follow_components([*dirs, str(tmp_path / "never")], set(SHAPES), {}, done=str(done))
    assert set(early) == set(dirs)
