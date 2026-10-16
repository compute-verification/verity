"""A TP Commit's serving rows (`pipeline.tp.serving_rows`, ``tp-commit --serving-rows``), on CPU.

* Ordering: `tp.commit.main` over a stand-in LLM of two ranks writes every rank's window after the ranks' finalize and before the
  challenge is drawn or any opening is asked for, and binds them to the TP run root of the ranks' roots.
* Fail closed: a rank without a window, a rank outside the world or a missing rank exits the Commit before its challenge; a Commit
  whose windows could not come before every draw is refused.
* A rank's window over real committed values: `write` on the served toy Commit (`tests.pipeline.served_rope`, its SiluMul rows: the
  "row's own Definition" port rule) is ``served``, its run the TP run root and the rank, its domains core's served domains over that
  run, and its hidden rows the single-GPU hook's (whose window, run the vllm-v1 root, is unchanged); two ranks' domains differ."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from verity_vllm.commit import serving_rows as CS
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.tp import serving_rows as SRT
from verity_vllm.pipeline.tp.commit import TP_ROOT_TAG, tp_run_root

WORKLOAD = {"requests": [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}], "sampling": {"max_tokens": 4}, "sha256": "w"}
RUN = {"wall_s": 0.5, "iterations": 4, "step_wall_s": [0.2, 0.1, 0.1, 0.1], "generated_tokens": 4, "prompt_tokens": 3, "tokens": {"q0": [5, 6, 7, 8]}}
ROOTS = {0: bytes(range(32)), 1: bytes(range(32, 64))}
SILU = "SiluMul_v1{I=32}"


class _Stop(Exception):
    pass


class _LLM:
    """`vllm.LLM` over two ranks: every `collective_rpc` is an event; `rank_rows` runs on a stand-in worker per rank, every other method
    answers one finalize-shaped dict per rank, and the challenge's openings (``tp2_commit_open``) end the run."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, **kw) -> _LLM:
        return self

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        name = getattr(method, "__name__", method)
        self.events.append(f"rpc.{name}")
        if name == "tp2_commit_open":
            raise _Stop
        if method is SRT.rank_rows:
            return [method(SimpleNamespace(_tp2_commit={"rank": r}), *args) for r in ROOTS]
        return [{"rank": r, "leaves": 3, "errors_n": 0, "errors": [], "committer": "native_collect_v2b", "run_root": ROOTS[r].hex(),
                 "leaves_per_step": [3]} for r in ROOTS]


def _tp_commit(tmp_path: Path, monkeypatch, write) -> tuple[list[str], Path]:
    """`tp.commit.main` with ``--serving-rows`` over `_LLM`, each rank's window `write`'s; -> the events and the request's dir."""
    pytest.importorskip("torch")
    from verity_vllm import protocol_options as PO
    from verity_vllm.commit import challenge, partition
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import commit as PC
    from verity_vllm.pipeline import gpu_lease as GL
    from verity_vllm.pipeline.tp import commit as TPCM

    events: list[str] = []
    vllm = ModuleType("vllm")
    vllm.LLM = _LLM(events)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    for owner, name, value in [(prof, "apply_env", lambda: {}), (prof, "checkpoint_for", lambda case, path: {"repo": "org/model"}),
                               (prof, "engine_kwargs", lambda cp, **kw: dict(kw)), (va, "load_workload", lambda path, max_tokens: WORKLOAD),
                               (va, "target_of_workload", lambda wl, *a: None), (va, "case_of", lambda wl, case: case),
                               (va, "throwaway_requests", lambda eng, reqs, seed=0: reqs), (PO, "commit_guard", lambda *a, **k: None),
                               (PC, "git_head", lambda: None), (PC, "run_workload", lambda eng, reqs, sampling: dict(RUN)),
                               (TPCM, "tree_of_record", lambda: {}), (GL, "_ENVIRON", []), (GL, "_HELD", []),
                               (partition, "require_partition", lambda doc, *a: {"program_digest_per_rank": {"0": "d0", "1": "d1"}, "ranks": 2}),
                               (challenge, "challenge_positions", lambda root, *a, **k: events.append(f"challenge {root.hex()}") or []),
                               (SRT, "write", write)]:
        monkeypatch.setattr(owner, name, value)
    builds = []
    for r in ROOTS:
        (b := tmp_path / f"build_rank{r}").mkdir()
        (b / "result.json").write_text(json.dumps({"program": {"digest": f"d{r}"}}))
        builds += ["--build-rank", str(b)]
    (tmp_path / "manifest.json").write_text("{}")
    req = tmp_path / "request.json"
    req.write_text(json.dumps({"dir": str(tmp_path / "serving-rows"), "partition": "partition.json", "run": "r-test", "committer": {"source": "t"}}))
    opts = cli.parse("tp-commit", ["--workload", str(tmp_path / "w.json"), "--out", str(tmp_path / "commit"), "--case", "C", "--pairs", "1",
                                   "--warmup", "1", "--only-arm", "instrumented", "--required-manifest", str(tmp_path / "manifest.json"), *builds,
                                   "--replay-deferred", "--serving-rows", str(req), "--label", "row-t"], environ={})
    with pytest.raises((_Stop, SystemExit)) as e:
        TPCM.main(opts)
    events.append(f"exit {e.value}" if e.type is SystemExit else "stopped at the openings")
    return events, tmp_path / "serving-rows"


def test_every_ranks_window_comes_after_every_root_and_before_the_challenge(tmp_path, monkeypatch):
    seen = []

    def write(st, request, tp):
        seen.append((st["rank"], request, tp))
        return {"run": SRT.run_of(bytes.fromhex(tp["tp_run_root"]), st["rank"]), "population": 8, "registration_digest": "ab" * 64}

    events, d = _tp_commit(tmp_path, monkeypatch, write)
    root = tp_run_root(2, ROOTS)
    last_finalize = max(i for i, e in enumerate(events) if e == "rpc.tp2_commit_finalize")
    assert events[last_finalize + 1:] == ["rpc.rank_rows", *[f"challenge {root.hex()}"] * 2, "rpc.tp2_commit_open", "stopped at the openings"]
    assert events.count("rpc.rank_rows") == 1
    tp = {"world": 2, "tp_run_root": root.hex(), "roots": {str(r): x.hex() for r, x in ROOTS.items()}, "root_tag": TP_ROOT_TAG.decode()}
    assert seen == [(r, {"dir": str(d), "partition": "partition.json", "run": "r-test", "committer": {"source": "t"}, "row": "row-t"}, tp) for r in ROOTS]
    doc = json.loads((d / "tp.json").read_text())
    assert (doc["result"], doc["tp_run_root"], doc["roots"]) == (True, root.hex(), tp["roots"])
    assert doc["runs"] == {"0": f"{root.hex()}/rank0", "1": f"{root.hex()}/rank1"} == {r: w["run"] for r, w in doc["windows"].items()}


def test_a_rank_that_cannot_write_its_window_exits_the_commit_before_its_challenge(tmp_path, monkeypatch):
    def write(st, request, tp):
        if st["rank"] == 1:
            raise LookupError("rank 1: no binding map")
        return {"run": SRT.run_of(bytes.fromhex(tp["tp_run_root"]), 0), "population": 8, "registration_digest": "ab" * 64}

    events, d = _tp_commit(tmp_path, monkeypatch, write)
    assert events[-2:] == ["rpc.rank_rows", "exit fail-closed: serving rows: rank 1: LookupError('rank 1: no binding map')"]
    assert not any(e.startswith("challenge") for e in events) and "rpc.tp2_commit_open" not in events
    doc = json.loads((d / "tp.json").read_text())
    assert doc["result"] is False and doc["windows"]["1"] is None and "no binding map" in (d / "rank1" / "error.txt").read_text()


@pytest.mark.parametrize("got, why", [
    ([{"rank": 0, "window": {}}], "rank 1: no window"),
    ([{"rank": 0, "window": {"population": 1}}, {"rank": 1, "error": "boom"}], "rank 1: boom"),
    ([{"rank": r, "window": {"population": 1}} for r in (0, 1, 2)], "rank 2 is not a rank of the world of 2"),
    ([{"rank": r, "window": {"population": 1}} for r in (0, 1, 1)], "3 answers for 2 ranks"),
], ids=["missing", "error", "extra", "duplicate"])
def test_the_driver_takes_every_rank_of_the_world_and_only_those(tmp_path, got, why):
    tp = {"world": 2, "tp_run_root": "00" * 32, "roots": {}, "root_tag": TP_ROOT_TAG.decode()}
    with pytest.raises(SystemExit, match="fail-closed: serving rows: .*" + why):
        SRT.gather(str(tmp_path), tp, got)
    assert json.loads((tmp_path / "tp.json").read_text())["result"] is False


@pytest.mark.parametrize("cfg, deferred, arm, refused", [
    ({}, "full", None, False), (None, "full", None, True), ({}, None, None, True), ({}, "full", "control", True)],
    ids=["deferred", "no-required-manifest", "in-process-replay", "control-only"])
def test_a_commit_whose_windows_could_not_come_before_every_draw_is_refused(cfg, deferred, arm, refused):
    args = SimpleNamespace(serving_rows="request.json", replay_deferred=deferred, only_arm=arm)
    if refused:
        with pytest.raises(SystemExit, match="--serving-rows on a TP Commit needs"):
            SRT.require(args, cfg)
    else:
        SRT.require(args, cfg)
    SRT.require(SimpleNamespace(serving_rows=None, replay_deferred=None, only_arm=None), None)


def test_a_ranks_window_over_its_committed_values_binds_the_tp_run_root(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import tests.check.test_sampled_replay as T
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.pipeline import deferred_rows as DRW
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline import serving_rows as SRW
    from tests.pipeline import served_rope

    monkeypatch.setattr(T, "I", 32)            # SiluMul's in0 (2I words) is one whole SHA-512 block
    c = served_rope.commit(tmp_path)
    (pf := tmp_path / "silu.json").write_text(json.dumps({"templates": [SILU], "law": "subset:1"}))
    r = DRW.of_record(RB.path(c["out"], 1), str(c["row_dir"]))
    bmap = json.loads((r.out / f"binding_map_p{r.pair}.json").read_text())
    weights = DRW.opened_weights(r)
    monkeypatch.setattr(SR, "model_weights_provider", lambda model, prefix: weights)
    rank_root = bytes(r.run.run_root)
    root = tp_run_root(1, {0: rank_root})
    reqs = [{"request_id": k, "arrive_step": v} for k, v in r.arrivals.items()]
    st = {"rank": 0, "rc": r.run, "com": r.store, "bmap": bmap, "eng": SimpleNamespace(model=None), "manifest_doc": None,
          "replay": {"programs": {"0": str(c["row_dir"] / "build_request")}, "of_record": {"0": [c["inst"]["digest"]]}, "requests": reqs}}
    tp = {"world": 1, "tp_run_root": root.hex(), "roots": {"0": rank_root.hex()}, "root_tag": TP_ROOT_TAG.decode()}
    req = {"dir": str(tmp_path / "tp"), "partition": str(pf), "run": "r-tp", "committer": {"source": "t"}, "row": "row-t"}
    got = SRT.write(st, req, tp)
    rec = json.loads((tmp_path / "tp" / "rank0" / "registration.json").read_text())
    w = rec["window"]
    assert (w["kind"], w["run"], got["run"], w["research_run"], w["row"]) == ("served", f"{root.hex()}/rank0", f"{root.hex()}/rank0", "r-tp", "row-t")
    assert w["tp"] == {"world": 1, "rank": 0, "tp_run_root": root.hex(), "run_root": rank_root.hex(), "root_tag": TP_ROOT_TAG.decode()}
    assert "of" not in w and "link" not in w and got["population"] == 8

    mine = dataclasses.replace(DRW.window_of_record(RB.path(c["out"], 1), str(c["row_dir"]), str(pf)), run=SRT.run_of(root, 0))
    assert {x["port"]: {"leaves": x["leaves"], "domain": x["domain"]} for x in rec["roots"]} == CS.expected_ports(mine) != \
        CS.expected_ports(dataclasses.replace(mine, run=SRT.run_of(root, 1)))
    assert not {p["domain"] for p in CS.expected_ports(mine).values()} & \
        {p["domain"] for p in CS.expected_ports(dataclasses.replace(mine, run=SRT.run_of(root, 1))).values()}

    # the single-GPU hook over the same members: its run the vllm-v1 root, its hidden rows the rank's
    (hreq := tmp_path / "request.json").write_text(json.dumps({"dir": str(tmp_path / "hook"), "partition": str(pf), "run": "r-hook"}))
    SRW.readers(str(hreq), r.store, r.run, weights, r.programs, bmap, r.arrivals, r.loaded, r.manifest, "row-t", r.digests)
    hook = json.loads((tmp_path / "hook" / "registration.json").read_text())
    assert (hook["window"]["kind"], hook["window"]["run"], "tp" in hook["window"]) == ("served", rank_root.hex(), False)
    n, words = 8, 64 + 64
    rows = [np.frombuffer((tmp_path / d / "m0-private.bin").read_bytes()[:n * words * 2], dtype="<u2") for d in ("tp/rank0", "hook")]
    assert np.array_equal(*rows) and rows[0].any()
