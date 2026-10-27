"""A TP Commit's serving rows (`pipeline.tp.serving_rows`, ``tp-commit --serving-rows``), on CPU.

* Ordering: `tp.commit.main` over a stand-in LLM of two ranks writes every rank's window after the ranks' finalize and before the
  challenge is drawn or any opening is asked for, and binds them to the TP run root of the ranks' roots.
* Fail closed: a rank without a window, a rank outside the world or a missing rank exits the Commit before its challenge; a Commit
  whose windows could not come before every draw is refused.
* A rank's window over real committed values: `write` on the served toy Commit (`tests.pipeline.served_rope`, its SiluMul rows: the
  "row's own Definition" port rule) is ``served``, its run the TP run root and the rank, its domains core's served domains over that
  run, and its hidden rows the single-GPU hook's (whose window, run the rank's own root, is unchanged); two ranks' domains differ.
* The verifier's side (`window_of_record`, `source`) on a TP Commit of two rows-sha512 ranks served on CPU under the SHA-512 TP run
  root (`tests.pipeline.served_tp`): each rank's registration passes, held to the verifier's own window of that rank from the
  Commit's record alone, and is refused for another rank, another TP run root, other served Programs, a stand-in, a record whose TP
  run root does not combine its ranks' roots (a 32-byte rank root by name), and a rank Program not of record; an old record's
  vllm-v1 ranks under the SHA-256 v1 root are refused as served by name (``sha512``).
* A deferred TP Commit that wrote no window (`served_tp --deferred`): each rank's ``stand-in`` from its sealed bundle (`from_bundle`)
  holds the values the in-Commit window of the same ranks holds, names the rank's rows-sha512 root under the TP run root and the
  bundle's manifest, and passes `source`; a stand-in naming another bundle, another link or another rank, or of a rank whose bundle
  the record does not name or whose replay failed, is refused, and so is a bundle that is not the record's."""
from __future__ import annotations

import dataclasses
import json
import multiprocessing
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from verity.ml.windows import rows as CS
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.tp import serving_rows as SRT
from verity_vllm.pipeline.tp import run_root as RT

from tests.pipeline import served_ranks

WORKLOAD = {"requests": [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}], "sampling": {"max_tokens": 4}, "sha256": "w",
            "execution": "bi-eager"}
RUN = {"wall_s": 0.5, "iterations": 4, "step_wall_s": [0.2, 0.1, 0.1, 0.1], "step_tokens_out": [1, 1, 1, 1], "generated_tokens": 4, "prompt_tokens": 3, "tokens": {"q0": [5, 6, 7, 8]}}
ROOTS = {0: bytes(range(64)), 1: bytes(range(64, 128))}
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
        if (served := served_ranks.answer(method, WORKLOAD, ROOTS)) is not None:
            return served
        return [{"rank": r, "leaves": 3, "errors_n": 0, "errors": [], "committer": "native_collect_v2b", "run_root": ROOTS[r].hex(),
                 "leaves_per_step": [3], "cpu_s": {}} for r in ROOTS]


def _tp_commit(tmp_path: Path, monkeypatch, write) -> tuple[list[str], Path]:
    """`tp.commit.main` with ``--serving-rows`` over `_LLM`, each rank's window `write`'s; -> the events and the request's dir."""
    pytest.importorskip("torch")
    from verity_vllm import protocols as PO
    from verity_vllm.commit import challenge, partition
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import commit as PC
    from verity_vllm.pipeline import gpu_lease as GL
    from verity_vllm.pipeline.tp import commit as TPCM

    events: list[str] = []
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
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
    root = RT.tp_run_root(2, ROOTS)
    last_finalize = max(i for i, e in enumerate(events) if e == "rpc.tp2_commit_finalize")
    assert events[last_finalize + 1:] == ["rpc.rank_rows", *[f"challenge {root.hex()}"] * 2, "rpc.tp2_commit_open", "stopped at the openings"]
    assert events.count("rpc.rank_rows") == 1
    tp = {"world": 2, "tp_run_root": root.hex(), "roots": {str(r): x.hex() for r, x in ROOTS.items()}, "root_tag": RT.TP_ROOT_TAG.decode()}
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
    tp = {"world": 2, "tp_run_root": "00" * 64, "roots": {}, "root_tag": RT.TP_ROOT_TAG.decode()}
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
    from verity_vllm.commit import step_rows
    from verity_vllm.pipeline import deferred_rows as DRW
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline import serving_rows as SRW
    from tests.pipeline import served_rope

    monkeypatch.setattr(T, "I", 32)            # SiluMul's in0 (2I words) is one whole SHA-512 block
    c = served_rope.commit(tmp_path, commit_scheme=step_rows.NAME)
    (pf := tmp_path / "silu.json").write_text(json.dumps({"templates": [SILU], "law": "subset:1"}))
    r = DRW.of_record(RB.path(c["out"], 1), str(c["row_dir"]))
    bmap = json.loads((r.out / f"binding_map_p{r.pair}.json").read_text())
    weights = DRW.opened_weights(r)
    monkeypatch.setattr(SR, "model_weights_provider", lambda model, prefix: weights)
    rank_root = bytes(r.run.run_root)
    root = RT.tp_run_root(1, {0: rank_root})
    reqs = [{"request_id": k, "arrive_step": v} for k, v in r.arrivals.items()]
    st = {"rank": 0, "rc": r.run, "com": r.store, "bmap": bmap, "eng": SimpleNamespace(model=None), "weights": None, "manifest_doc": None,
          "replay": {"programs": {"0": str(c["row_dir"] / "build_request")}, "of_record": {"0": [c["inst"]["digest"]]}, "requests": reqs}}
    tp = {"world": 1, "tp_run_root": root.hex(), "roots": {"0": rank_root.hex()}, "root_tag": RT.TP_ROOT_TAG.decode()}
    req = {"dir": str(tmp_path / "tp"), "partition": str(pf), "run": "r-tp", "committer": {"source": "t"}, "row": "row-t"}
    ctx = multiprocessing.get_context("fork")
    q = ctx.SimpleQueue()

    def on_a_rank():
        try:
            q.put(SRT.write(st, req, tp))
        except BaseException as e:  # noqa: BLE001 -- reported to the test, which fails on it
            q.put(repr(e))

    p = ctx.Process(target=on_a_rank, daemon=True)     # as vLLM's TP workers run: a daemonic process, which can fork no pool
    p.start()
    got = q.get()
    p.join()
    assert isinstance(got, dict), got
    rec = json.loads((tmp_path / "tp" / "rank0" / "registration.json").read_text())
    w = rec["window"]
    assert (w["kind"], w["run"], got["run"], w["research_run"], w["row"]) == ("served", f"{root.hex()}/rank0", f"{root.hex()}/rank0", "r-tp", "row-t")
    assert w["tp"] == {"world": 1, "rank": 0, "tp_run_root": root.hex(), "run_root": rank_root.hex(), "root_tag": RT.TP_ROOT_TAG.decode()}
    assert "of" not in w and "link" not in w and got["population"] == 8 and w["commitment"] == step_rows.NAME

    mine = dataclasses.replace(DRW.window_of_record(RB.path(c["out"], 1), str(c["row_dir"]), str(pf)), run=SRT.run_of(root, 0))
    assert {x["port"]: {"leaves": x["leaves"], "domain": x["domain"]} for x in rec["roots"]} == CS.expected_ports(mine) != \
        CS.expected_ports(dataclasses.replace(mine, run=SRT.run_of(root, 1)))
    assert not {p["domain"] for p in CS.expected_ports(mine).values()} & \
        {p["domain"] for p in CS.expected_ports(dataclasses.replace(mine, run=SRT.run_of(root, 1))).values()}

    # the single-GPU hook over the same members: its run the rank's own root, its hidden rows the rank's
    (hreq := tmp_path / "request.json").write_text(json.dumps({"dir": str(tmp_path / "hook"), "partition": str(pf), "run": "r-hook"}))
    SRW.readers(str(hreq), r.store, r.run, weights, r.programs, bmap, r.arrivals, r.loaded, r.manifest, "row-t", r.digests)
    hook = json.loads((tmp_path / "hook" / "registration.json").read_text())
    assert (hook["window"]["kind"], hook["window"]["run"], "tp" in hook["window"]) == ("served", rank_root.hex(), False)
    n, words = 8, 64 + 64
    rows = [np.frombuffer((tmp_path / d / "m0-private.bin").read_bytes()[:n * words * 2], dtype="<u2") for d in ("tp/rank0", "hook")]
    assert np.array_equal(*rows) and rows[0].any()


@pytest.fixture(scope="module")
def served_tp(tmp_path_factory):
    pytest.importorskip("torch")
    from tests.pipeline import served_tp as ST

    return ST.commit(tmp_path_factory.mktemp("served-tp"), world=2, I=256, law="subset:2")


@pytest.fixture(scope="module")
def served_tp_v1(tmp_path_factory):
    """An old record's TP Commit: vllm-v1 ranks under the SHA-256 ``verity/tp-run-root/v1``."""
    pytest.importorskip("torch")
    from tests.pipeline import served_tp as ST

    return ST.commit(tmp_path_factory.mktemp("served-tp-v1"), world=2, I=256, law="subset:2", commit_scheme="vllm-v1")


def _registered(c, rank: int) -> dict:
    return json.loads((c["windows"] / f"rank{rank}" / "registration.json").read_text())


def _mine(c, rank: int, commit_dir=None):
    return SRT.window_of_record(commit_dir or c["commit"], rank, c["partition"], c["build"], c["workload"])


def test_each_ranks_registration_is_held_to_the_verifiers_window_of_that_rank(served_tp):
    c = served_tp
    for r in range(2):
        mine, held = _mine(c, r)
        rec = _registered(c, r)
        assert mine.run == f"{c['tp_run_root']}/rank{r}" and mine.law == {"law": "subset", "k": 2}
        assert [(m.template, m.n) for m in mine.members] == [("SiluMul_v2{I=256}", 8)]
        assert SRT.source(rec["window"], mine, held) == {"ok": True, "codes": [], "detail": {"kind": "served", "run": mine.run, "tp": held["tp"]}}
        assert {x["port"]: {"leaves": x["leaves"], "domain": x["domain"]} for x in rec["roots"]} == CS.expected_ports(mine)
        assert held["tp"]["run_root"] == SRT.record(c["commit"])["roots"][str(r)]
    other, held1 = _mine(c, 1)
    assert SRT.source(_registered(c, 0)["window"], other, held1)["codes"] == ["run", "tp"]


def test_a_ranks_served_window_under_a_sha512_tp_run_root_passes(served_tp):
    rec = SRT.record(served_tp["commit"])
    assert rec["root_tag"] == RT.TP_ROOT_TAG.decode() and len(bytes.fromhex(rec["tp_run_root"])) == 64
    assert all(len(bytes.fromhex(x)) == 64 for x in rec["roots"].values())
    for r in range(2):
        w, (mine, held) = _registered(served_tp, r), _mine(served_tp, r)
        assert w["window"]["run"] == mine.run and w["window"]["commitment"] == "rows-sha512"
        assert SRT.source(w["window"], mine, held) == {"ok": True, "codes": [], "detail": {"kind": "served", "run": mine.run, "tp": held["tp"]}}


def test_a_ranks_served_window_under_an_old_records_sha256_tp_run_root_is_refused_by_name(served_tp_v1):
    """A served window names a SHA-512 commitment, as `deferred_rows.source` holds a deferred Commit's (one hash): an old record's
    windows, vllm-v1 ranks under the SHA-256 v1 TP run root and held to the verifier's on everything else, are refused by ``sha512``."""
    assert SRT.record(served_tp_v1["commit"])["root_tag"] == RT.TP_ROOT_TAG_V1.decode()
    for r in range(2):
        mine, held = _mine(served_tp_v1, r)
        assert len(bytes.fromhex(held["tp"]["tp_run_root"])) == len(bytes.fromhex(held["tp"]["run_root"])) == 32
        got = SRT.source(_registered(served_tp_v1, r)["window"], mine, held)
        assert (got["ok"], got["codes"]) == (False, ["sha512"])


@pytest.mark.parametrize("alter, codes", [
    (lambda w: {**w, "kind": "stand-in", "of": {"run_root": "00" * 64}, "link": "verity-vllm serving-rows"}, ["kind", "no_link"]),
    (lambda w: {**w, "run": "00" * 64 + "/rank0", "tp": {**w["tp"], "tp_run_root": "00" * 64}}, ["run", "tp"]),
    (lambda w: {**w, "tp": {**w["tp"], "run_root": "00" * 64}}, ["tp"]),
    (lambda w: {**w, "served_programs": {**w["served_programs"], "r0": "00" * 32}}, ["served_programs"]),
    (lambda w: None, ["kind", "run", "tp", "served_programs"]),
], ids=["stand-in", "other-tp-run-root", "other-rank-root", "other-served-program", "no-window"])
def test_a_registration_that_is_not_the_ranks_served_window_of_record_is_refused(served_tp, alter, codes):
    mine, held = _mine(served_tp, 0)
    got = SRT.source(alter(_registered(served_tp, 0)["window"]), mine, held)
    assert (got["ok"], got["codes"]) == (False, codes)


def _copy_commit(c, tmp_path: Path, runs=None, versions=None) -> Path:
    d = tmp_path / "commit"
    d.mkdir()
    rows = [json.loads(x) for x in (c["commit"] / "runs.jsonl").read_text().splitlines()]
    v = json.loads((c["commit"] / "versions.json").read_text())
    (d / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (runs(rows) if runs else rows)))
    (d / "versions.json").write_text(json.dumps(versions(v) if versions else v))
    return d


def _rank_root(rows, rank, root):
    rows[1]["commit"]["per_rank"][str(rank)]["run_root"] = root
    return rows


@pytest.mark.parametrize("runs, versions, why", [
    (lambda rows: _rank_root(rows, 1, "00" * 64), None, "does not combine its ranks' roots"),
    (lambda rows: _rank_root(rows, 1, "00" * 32), None, "rank 1 root is 32 bytes, a vllm-v1"),
    (lambda rows: [rows[1] | {"pair": 1}], None, "no pair-0 instrumented run"),
    (lambda rows: [rows[0]], None, "no pair-0 instrumented run"),
    (lambda rows: [rows[1] | {"commit": {**rows[1]["commit"], "world": 3}}], None, "are not the world of 3"),
    (None, lambda v: {**v, "root_tag": "verity/tp-run-root/v0"}, "root tag 'verity/tp-run-root/v0' is not a TP run root form"),
    (None, lambda v: {**v, "root_tag": RT.TP_ROOT_TAG_V1.decode()}, "rank 0 root is not 32 bytes"),
    (None, lambda v: {**v, "sampled_replay": {"of_record": {"0": ["00" * 32], "1": v["sampled_replay"]["of_record"]["1"]}}},
     "REFUSED \\(not the Program of record\\)"),
], ids=["root-not-combined", "rank-root-32-bytes", "pair-1-only", "control-only", "world-not-the-ranks", "root-tag", "root-tag-v1",
        "program-not-of-record"])
def test_the_verifiers_window_is_refused_on_a_record_it_cannot_hold(served_tp, tmp_path, runs, versions, why):
    d = _copy_commit(served_tp, tmp_path, runs, versions)
    with pytest.raises((LookupError, ValueError, RuntimeError), match=why):
        _mine(served_tp, 0, d)


def test_the_record_is_the_last_commits_into_the_same_out(served_tp, tmp_path):
    def earlier(rows):
        old = json.loads(json.dumps(rows[1]))
        old["commit"]["per_rank"]["0"]["run_root"] = "11" * 64
        old["commit"]["tp_run_root"] = RT.tp_run_root(2, {r: bytes.fromhex(x["run_root"]) for r, x in
                                                       ((int(k), v) for k, v in old["commit"]["per_rank"].items())}).hex()
        return [rows[0], old, old | {"pair": 1}, *rows]

    d = _copy_commit(served_tp, tmp_path, earlier)
    assert SRT.record(d)["tp_run_root"] == served_tp["tp_run_root"]
    mine, held = _mine(served_tp, 0, d)
    assert SRT.source(_registered(served_tp, 0)["window"], mine, held)["ok"]


def test_a_rank_outside_the_world_has_no_window(served_tp):
    with pytest.raises(ValueError, match="rank 2 is not a rank of the world of 2"):
        _mine(served_tp, 2)


def test_the_verifiers_window_is_refused_for_a_workload_the_record_did_not_serve(served_tp, tmp_path):
    doc = json.loads(served_tp["workload"].read_text())
    doc["requests"][0]["max_tokens"] += 1
    (wl := tmp_path / "workload.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="workload sha256 .* is not the record's"):
        SRT.window_of_record(served_tp["commit"], 0, served_tp["partition"], served_tp["build"], wl)
    with pytest.raises(ValueError, match="is not the record's None"):
        _mine(served_tp, 0, _copy_commit(served_tp, tmp_path, lambda rows: [{k: v for k, v in x.items() if k != "workload_sha256"} for x in rows]))


@pytest.fixture(scope="module")
def deferred_tp(tmp_path_factory):
    """The TP Commit of `served_tp`'s ranks with no window and every rank's replay deferred to its sealed bundle, and each rank's
    stand-in window from that bundle."""
    pytest.importorskip("torch")
    from tests.pipeline import served_tp as ST

    root = tmp_path_factory.mktemp("deferred-tp")
    c = ST.commit(root / "tp", world=2, I=256, law="subset:2", deferred=True)
    c["windows"] = root / "stand-in"
    c["made"] = {r: SRT.from_bundle(c["commit"], r, c["partition"], c["windows"] / f"rank{r}", workers=1) for r in range(2)}
    return c


def _values(window_dir: Path) -> bytes:
    """Member 0's committed rows, its prover body less the salts."""
    m = json.loads((window_dir / "window.json").read_text())["members"][0]
    body = (window_dir / "m0-private.bin").read_bytes()
    return body[:len(body) - m["n"] * len(m["ports"]) * CS.SALT_BYTES]


def _bundle_sha(c, rank: int) -> str:
    return next(b["manifest_sha256"] for b in json.loads((c["commit"] / "replay_deferred.json").read_text())["bundles"] if b["rank"] == rank)


def test_a_deferred_ranks_stand_in_from_its_bundle_holds_the_served_values_and_passes(deferred_tp, served_tp):
    c = deferred_tp
    assert c["tp"] is None and not (c["commit"].parent / "serving-rows").exists()
    for r in range(2):
        mine, held = _mine(c, r)
        rec = _registered(c, r)
        w = rec["window"]
        assert (w["kind"], w["run"], c["made"][r]["run"], w["commitment"], w["tp"]) == ("stand-in", mine.run, mine.run, "rows-sha512", held["tp"])
        assert w["of"] == {"commitment": "rows-sha512", "tp_run_root": c["tp_run_root"], "rank": r, "run_root": held["tp"]["run_root"],
                           "bundle_manifest_sha256": _bundle_sha(c, r), "rows": SRT.ROWS} and w["link"] == SRT.LINK
        assert SRT.source(w, mine, held) == {"ok": True, "codes": [], "detail": {"kind": "stand-in", "run": mine.run, "tp": held["tp"], "of": w["of"]}}
        assert {x["port"]: {"leaves": x["leaves"], "domain": x["domain"]} for x in rec["roots"]} == CS.expected_ports(mine)
        assert c["made"][r]["population"] == 8 and _values(c["windows"] / f"rank{r}") == _values(served_tp["windows"] / f"rank{r}")
    assert _values(c["windows"] / "rank0") != _values(c["windows"] / "rank1")


def test_the_verifiers_window_and_a_stand_in_index_the_ranks_program_once(deferred_tp, tmp_path, monkeypatch):
    """`window_of_record` and `from_bundle` each index the rank's Program once: their population takes the index `rank_replay.load`
    built for every request served by that Program (here both requests, which share the rank's one Program dir), and gives the
    window, its units and its members that indexing each request's Program again gives."""
    from verity_vllm.check.replay import index as IX
    from verity_vllm.pipeline import serving_rows as SRW

    c, built, init = deferred_tp, [], IX.ProgramIndex.__init__
    requests = len(json.loads(c["workload"].read_text())["requests"])

    def counted(self, inst, **kw):
        built.append(kw)
        init(self, inst, **kw)

    def made(d):
        (mine, held), res = _mine(c, 0), SRT.from_bundle(c["commit"], 0, c["partition"], d, workers=1)
        return ((mine.run, mine.law, mine.query, mine.members, mine.row_tables, mine.partition_digest, held),
                {k: res[k] for k in ("run", "population", "partition_digest")},
                [(d / f).read_bytes() for f in ("partition.json", "index.json", "m0-units.json")], _values(d))

    monkeypatch.setattr(IX.ProgramIndex, "__init__", counted)
    once = made(tmp_path / "once")
    assert len(built) == 2
    monkeypatch.setattr(SRW, "index_of", lambda inst, reuse=(), **kw: IX.ProgramIndex(inst, **kw))
    assert made(tmp_path / "again") == once and requests == 2 and len(built) == 2 + 2 * (1 + requests)
    assert once[3] == _values(c["windows"] / "rank0")


def _pending(rows, rank, commit_pass):
    rows[1]["commit"]["per_rank"][str(rank)]["commit_pass"] = commit_pass
    return rows


def _copy_deferred(c, tmp_path: Path, runs=None) -> Path:
    d = _copy_commit(c, tmp_path, runs)
    (d / "replay_deferred.json").write_text((c["commit"] / "replay_deferred.json").read_text())
    return d


@pytest.mark.parametrize("alter, record, codes", [
    (lambda w: {**w, "of": {**w["of"], "bundle_manifest_sha256": "00" * 32}}, None, ["of"]),
    (lambda w: {**w, "link": "verity-vllm serving-rows"}, None, ["link"]),
    (lambda w: {**w, "of": {**w["of"], "commitment": "vllm-v1"}}, None, ["of"]),
    (lambda w: w, lambda c, t: _copy_commit(c, t), ["kind", "no_link"]),
    (lambda w: w, lambda c, t: _copy_deferred(c, t, lambda rows: _pending(rows, 0, False)), ["kind", "no_link"]),
    (lambda w: {k: v for k, v in w.items() if k not in ("of", "link")}, None, ["of", "link"]),
], ids=["other-bundle", "other-link", "other-commitment", "record-names-no-bundle", "replay-failed", "stand-in-without-link"])
def test_a_stand_in_that_is_not_the_ranks_bundle_of_record_is_refused(deferred_tp, tmp_path, alter, record, codes):
    mine, held = _mine(deferred_tp, 0, record(deferred_tp, tmp_path) if record else None)
    got = SRT.source(alter(_registered(deferred_tp, 0)["window"]), mine, held)
    assert (got["ok"], got["codes"]) == (False, codes)


def test_another_ranks_stand_in_and_a_decided_records_are_held_by_rank(deferred_tp, tmp_path):
    mine1, held1 = _mine(deferred_tp, 1)
    assert SRT.source(_registered(deferred_tp, 0)["window"], mine1, held1)["codes"] == ["run", "tp", "of"]
    d = _copy_deferred(deferred_tp, tmp_path, lambda rows: _pending(_pending(rows, 0, True), 1, False))
    assert SRT.record(d)["bundles"] == {"0": _bundle_sha(deferred_tp, 0)}
    mine, held = _mine(deferred_tp, 0, d)
    assert SRT.source(_registered(deferred_tp, 0)["window"], mine, held)["ok"], "a rank whose replay passed keeps its stand-in"


@pytest.mark.parametrize("tamper, why", [
    (lambda d: (d / "replay_deferred.json").unlink(), "not a deferred TP Commit"),
    (lambda d: (d / "replay_bundle_p0_rank1" / "binding_map.json").write_text("{}"), "sha256 is not the manifest's"),
    (lambda d: _tamper(d, lambda rows: _pending(rows, 1, False)), "no rank whose deferred replay is pending or passed"),
    (lambda d: _tamper(d, lambda rows: _rank_root(rows, 1, "11" * 64)), "commitment vector|run root of record"),
], ids=["not-deferred", "bundle-altered", "replay-failed", "other-rank-root"])
def test_a_bundle_that_is_not_the_records_gives_no_stand_in(deferred_tp, tmp_path, tamper, why):
    from verity_vllm.pipeline.tp import commit_replay as CR

    d = tmp_path / "commit"
    shutil.copytree(deferred_tp["commit"], d)
    tamper(d)
    with pytest.raises(CR.Refused, match=why):
        SRT.from_bundle(d, 1, deferred_tp["partition"], tmp_path / "w", workers=1)
    assert not (tmp_path / "w").exists()


def _tamper(d: Path, fn) -> None:
    rows = fn([json.loads(x) for x in (d / "runs.jsonl").read_text().splitlines()])
    (d / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def test_the_command_writes_each_ranks_stand_in_and_refuses_a_commit_that_deferred_nothing(deferred_tp, served_tp, tmp_path, capsys):
    o = cli.parse("tp-serving-rows", [str(deferred_tp["commit"]), "--partition", str(deferred_tp["partition"]), "--out", str(tmp_path / "w"),
                                      "--ranks", "1", "--workers", "1"], environ={})
    assert SRT.main(o) == 0 and sorted(p.name for p in (tmp_path / "w").iterdir()) == ["rank1"]
    mine, held = _mine(deferred_tp, 1)
    assert SRT.source(json.loads((tmp_path / "w" / "rank1" / "registration.json").read_text())["window"], mine, held)["ok"]
    o = cli.parse("tp-serving-rows", [str(served_tp["commit"]), "--partition", str(served_tp["partition"]), "--out", str(tmp_path / "v")], environ={})
    assert SRT.main(o) == 2 and "REFUSED" in capsys.readouterr().out and not (tmp_path / "v").exists()
