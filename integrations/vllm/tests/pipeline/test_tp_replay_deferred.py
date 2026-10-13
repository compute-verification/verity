"""A deferred TP Commit (`tp-commit --replay-deferred`) and its CPU replay (`pipeline.tp.commit_replay`): every rank's C2 decided
from its sealed bundle exactly as the rank's process would have decided it in place (the same gates, the same components, the
same row verdict), a bundle that is not the record's refused by name, the rank's shards composed at (world, rank), and the row's
pass-through (`row_tp`)."""
from __future__ import annotations

import copy
import gzip
import hashlib
import itertools
import json
import types
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from verity_vllm.commit import scheme
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline.research_outputs import KEPT_APART
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.pipeline.tp import commit as TC
from verity_vllm.pipeline.tp import commit_replay as TPR
from verity_vllm.pipeline.tp import rank_replay as RR

WORLD = 2
SEED = 7
REPLAY_CFG = {"programs": {"0": "/b/rank0", "1": "/b/rank1"}, "of_record": {"0": ["a" * 64], "1": ["b" * 64]}, "requests": [],
              "per_stratum": 2, "max_evaluate": 2000, "seed": SEED, "workers": None, "challenge": None, "targets": [], "ring_slots": None}
VERSIONS = {"sampled_replay": {"of_record": REPLAY_CFG["of_record"], "seed": SEED}}


def _sr(result=True, grade="complete", *, rank=0, qp_ok=True, mismatch=0):
    """A rank's sampled-replay record as `driver.sampled_replay` writes it (population reconciled against Q(P) unless qp_ok is not True)."""
    pop = {"vus": 10, "strata_n": 3, "source": "rows", "by_family": {"Gemm_v1": 10}, "query_total": {"identities": 10}}
    if qp_ok is not None:
        pop["query_checks"] = {"ok": qp_ok, **({} if qp_ok else {"vus_outside_query": {"n": 2, "first": [{"request_id": "r0", "step": 0, "op_path": "x", "member": "y"}]}})}
    return {"result": result, "grade": grade, "rank": rank, "why": None if result else f"rank {rank}: {grade}",
            "population": pop, "sample": {"picked": 4, "evaluated": 4, "equal": 4 - mismatch, "mismatch_n": mismatch, "not_evaluated_n": 0,
                                          "per_stratum": 2, "seed": SEED, "seed_source": "--replay-seed (targeted)"},
            "strata_keys": {"mechanisms": ["Gemm_v1"], "requests": ["r0"], "phases": ["prefill"], "steps": {"early": [0], "late": [1]},
                            "lifetime": {"values": {"fresh-window": 10}, "ring_slots": 8}, "key": "spec | request | rank"},
            "unreplayable": [{"family": "AllReduce2_v1", "n": 3, "why": "no registered evaluator"}], "weights": {"equal": True},
            "mismatch": [{"i": 1, "op_path": "model.layers.0.mlp"}] * mismatch, "strata": {"a": {}}, "trace": "t"}


def _lk(result=True, rank=0):
    return {"result": result, "rank": rank, "pairs": 9, "linked": 9 if result else 8, "why": None if result else f"rank {rank}: step 3 unlinked"}


def _fin(rank, sr, lk, pin=True):
    return {"rank": rank, "sampled_replay": {"schema": RR.SCHEMA, "rank": rank, "method": RR.METHOD, "sampled_replay": sr, "boundary_linkage": lk},
            "weights_pin": {"result": pin, "why": None if pin else "1 served weight(s) differ", "mismatched": [] if pin else [{"served_name": "model.x.weight"}]}}


def _deferred_fin(rank, out=Path("/o"), pin=True):
    sr, lk = RB.deferred_records(RB.path(out, 0, rank=rank))
    return _fin(rank, dict(sr, rank=rank), dict(lk, rank=rank), pin)


# ------------------------------------------------------------------------------------------------ the bundle of a rank

def test_a_tp_ranks_bundle_is_its_own_and_never_published():
    assert RB.path("/o", 1).name == "replay_bundle_p1" and RB.partial("/o", 1).name == "replay_bundle_p1.partial"
    assert RB.path("/o", 1, rank=3).name == "replay_bundle_p1_rank3" and RB.partial("/o", 1, rank=3).name == "replay_bundle_p1_rank3.partial"
    assert RB.path("/o", 1, rank=3).name.startswith(KEPT_APART) and TPR.result_path("/o", 1, 3).name == "sampled_replay_p1_rank3.json"


CHUNK = 16
RNG = np.random.default_rng(11)
PARTS = {"model.layers.0.self_attn.q_proj.weight": RNG.integers(0, 1 << 16, (4, 8), dtype=np.uint16),
         "model.layers.0.self_attn.k_proj.weight": RNG.integers(0, 1 << 16, (2, 8), dtype=np.uint16),
         "model.layers.0.self_attn.v_proj.weight": RNG.integers(0, 1 << 16, (2, 8), dtype=np.uint16),
         "model.layers.0.mlp.down_proj.weight": RNG.integers(0, 1 << 16, (8, 6), dtype=np.uint16),
         "model.norm.weight": RNG.integers(0, 1 << 16, (8,), dtype=np.uint16)}
QKV = "model.layers.0.self_attn.qkv_proj.weight"


def _rank_shards(rank):
    """Rank ``rank``'s registered tensors of a TP2 engine, sliced here by hand: q|k|v per part on dim 0, down_proj on dim 1."""
    q, k, v = (PARTS[f"model.layers.0.self_attn.{p}_proj.weight"] for p in "qkv")
    qkv = np.concatenate([q[2 * rank:2 * rank + 2], k[rank:rank + 1], v[rank:rank + 1]])
    down = PARTS["model.layers.0.mlp.down_proj.weight"][:, 3 * rank:3 * rank + 3]
    return {QKV: qkv, "model.layers.0.mlp.down_proj.weight": down, "model.norm.weight": PARTS["model.norm.weight"]}


@pytest.fixture
def rank_weights(tmp_path):
    from tests.pipeline.test_cpu_replay import _shard

    ckd = tmp_path / "ck"
    ckd.mkdir()
    (ckd / "config.json").write_text("{}")
    _shard(ckd / "model.safetensors", PARTS)

    def tree(rank):
        names, roots = [], []
        for name, a in _rank_shards(rank).items():
            raw = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
            names.append([name, "bfloat16", list(a.shape), int(raw.size)])
            roots.append(CPU.tensor_root(raw, CHUNK))
        geo = bytes(32)
        return {"names": names, "tensor_roots": [r.hex() for r in roots], "root": scheme.weights_root(geo, scheme.names_digest(names), roots).hex(),
                "chunk": CHUNK, "geo_digest": geo.hex(), "extra": {}}

    return ckd, tree


def test_a_ranks_shards_open_against_its_committed_root_and_another_ranks_do_not(tmp_path, rank_weights):
    ckd, tree = rank_weights
    ck = WR.CheckpointIndex(str(ckd), verify_shards=False)
    t1 = tree(1)
    words, wc = CPU.open_weights(tmp_path, t1, t1["root"], ck, threads=2, tp={"world": WORLD, "rank": 1})
    assert wc["equal"] is True and wc["opened"] == 3 and wc["unopened_n"] == 0, wc
    assert np.array_equal(words[QKV].reshape(-1), _rank_shards(1)[QKV].reshape(-1))
    _, wc0 = CPU.open_weights(tmp_path, t1, t1["root"], ck, threads=2, tp={"world": WORLD, "rank": 0})
    assert wc0["equal"] is False and sorted(m["name"] for m in wc0["mismatched"]) == sorted([QKV, "model.layers.0.mlp.down_proj.weight"]), \
        "rank 0's shards are not rank 1's committed tensors; the replicated norm is"
    _, wc_full = CPU.open_weights(tmp_path, t1, t1["root"], ck, threads=2)
    assert wc_full["opened"] == 1 and {u["name"] for u in wc_full["unopened"]} == {QKV, "model.layers.0.mlp.down_proj.weight"}, \
        "composed as the whole tensor, a shard is not a tensor of the checkpoint"


# ------------------------------------------------------------------------------------------------ the gates: in place == deferred + fold

CASES = list(itertools.product(
    [(True, None), (False, "identity coverage: 2 of 9 required identities have no binding")],
    [(True, "complete", 0), (False, "mismatch", 1), (None, "not-evaluated", 0)],
    [True, False],
    [True, False, None],
    [True, False, None]))


@pytest.mark.parametrize("pre,replay,linked,qp_ok,pin", CASES)
def test_a_deferred_ranks_verdict_after_its_fold_is_the_in_process_verdict(pre, replay, linked, qp_ok, pin):
    rank = 1
    result, grade, mismatch = replay
    sr, lk = _sr(result, grade, rank=rank, qp_ok=qp_ok, mismatch=mismatch), _lk(linked, rank)
    here, there = {}, {}
    ok_here, why_here = C2.c2_and_pin_gates(here, _fin(rank, copy.deepcopy(sr), copy.deepcopy(lk), pin), *pre, rank=rank, replay_on=True, required=True, bundle="b")
    ok, why = C2.c2_and_pin_gates(there, _deferred_fin(rank, pin=pin), *pre, rank=rank, replay_on=True, required=True, bundle="b")
    if ok is not None:
        assert ok is False and ok_here is False, "a rank decided before its replay fails in place too"
        assert pre[0] is False or pin is not True
        return
    assert (why, there["replay_deferred"]) == (RB.PENDING, {"pre": list(pre), "bundle": "b"}) and pre[0] and pin is True
    got = C2.replay_gates(there, copy.deepcopy(sr), copy.deepcopy(lk), *there["replay_deferred"]["pre"], rank=rank, replay_on=True, required=True)
    assert got == (ok_here, why_here)
    for k in ("sampled_replay", "boundary_linkage", "query_population", "weights_pin"):
        assert there[k] == here[k], k


def test_without_a_replay_of_record_the_gates_are_the_pin_alone():
    cb = {}
    ok, why = C2.c2_and_pin_gates(cb, _fin(0, {}, {}, True), True, None, rank=0, replay_on=False, required=True, bundle="b")
    assert (ok, why) == (True, None) and "query_population" not in cb and "replay_deferred" not in cb


# ------------------------------------------------------------------------------------------------ a deferred Commit on disk

def _bmap(rank, root):
    entries = [{"identity": {"rank": rank, "op": "model.embed_tokens"}, "kind": "leaf", "step_root": "00", "leaf_lo": 0, "leaf_hi": 1}]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return {"schema": "m", "run_root": root, "rank": rank, "entries_n": 1, "digest": digest, "entries": entries}


def _cblock(rank, root, bmap):
    return {"run_root": root, "steps": 2, "leaves": 3, "bytes_bound": 516, "tensors": 2, "openings": {"n": 64, "verified": 64},
            "classes": {}, "weight_registration": {"root": "ab" * 32}, "binding": {"map_digest": bmap["digest"], "coverage": {"ok": True}}}


def _in_process(fins):
    """The Commit's rows, components and summary fields as the in-process path derives them from the ranks' finalize records."""
    per_rank, reasons = {}, []
    for f in fins:
        cb = _cblock(f["rank"], "r", {"digest": "d"})
        ok, why = C2.c2_and_pin_gates(cb, f, True, None, rank=f["rank"], replay_on=True, required=True, bundle="b")
        cb["commit_pass"], cb["commit_fail_reason"] = ok, why
        per_rank[f["rank"]] = cb
        if ok is False:
            reasons.append(f"rank {f['rank']}: {why}")
    rows = [{"pair": 0, "arm": "instrumented", "tokens_equal_ref": True, "commit": {"per_rank": per_rank, "commit_pass": not reasons,
                                                                                  "commit_fail_reason": "PASS" if not reasons else "; ".join(reasons)}}]
    sr_comp, lk_comp = C2.sampled_replay_components(rows, VERSIONS)
    return json.loads(json.dumps({"row": rows[0]["commit"], "sampled_replay": {k: sr_comp.get(k) for k in C2.SR_SUMMARY_KEYS}, "boundary_linkage": lk_comp,
                                  "query_population": C2.query_population_component(rows, True)}, default=str))


@pytest.fixture
def deferred(tmp_path):
    """A one-run TP2 Commit (the config run's only-arm form) as `tp-commit --replay-deferred` leaves it: every rank PENDING, its bundle
    sealed (a real dumped store of one committed run, its binding map, its weights tree), `replay_deferred.json` naming them."""
    pytest.importorskip("torch")
    from tests.check.opened import commit_steps
    from verity_vllm.commit import store_dump as SD

    out = tmp_path / "commit"
    out.mkdir()
    com, rc = commit_steps({0: [("model.embed_tokens", bytes(range(256)) * 2)], 1: [("runner.sampler", b"\x01\x02\x03\x04")]})
    root = rc.run_root.hex()
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"identities": [], "ranks": WORLD}))
    per_rank, fins, bundles_of = {}, [], {0: {}}
    for r in range(WORLD):
        bmap = _bmap(r, root)
        part = RB.partial(out, 0, rank=r)
        SD.dump(com, rc, part / "store")
        (part / "weights").mkdir()
        (part / "weights" / "tree.json").write_text(json.dumps({"names": [], "tensor_roots": [], "root": "ab" * 32, "chunk": CHUNK, "extra": {}}))
        (part / "binding_map.json").write_text(json.dumps(bmap))
        f = _deferred_fin(r, out)
        cb = _cblock(r, root, bmap)
        ok, why = C2.c2_and_pin_gates(cb, f, True, None, rank=r, replay_on=True, required=True, bundle=RB.path(out, 0, rank=r).name)
        cb["commit_pass"], cb["commit_fail_reason"] = ok, why
        per_rank[r], bundles_of[0][r] = cb, {"partial": str(part), "hashed": {}, "engine_of_record": {"vllm": "test"}, "programs": {}}
        fins.append({k: v for k, v in f.items()})
    vec = {r: rc.run_root for r in range(WORLD)}
    rows = [{"pair": 0, "arm": "instrumented", "tokens_equal_ref": True, "wall_s": 1.0,
             "commit": {"tp_run_root": TC.tp_run_root(WORLD, vec).hex(), "world": WORLD, "per_rank": per_rank,
                        "commitment_vector": [{"rank": r, "run_root": root} for r in range(WORLD)], "commit_pass": None, "commit_fail_reason": RB.PENDING}}]
    sr_comp, lk_comp = C2.sampled_replay_components(rows, VERSIONS)
    qp_comp = C2.query_population_component(rows, True)
    comps = {"weights_pin": {"result": True}, "cross_rank_collectives": {"result": True}, "fold_match_binding": {"result": None, "why": "no --match-dir"}}
    vc = C2.annotate_value_correspondence(None, sr_comp, lk_comp, qp_comp, comps["weights_pin"], comps["cross_rank_collectives"], comps["fold_match_binding"], replay_on=True)
    assert sr_comp["result"] is None and qp_comp["result"] is None and vc["result"] is None and vc["c2_of_record"]["result"] is None
    summ = {"label": "t", "tp": WORLD, "pairs": 1, "commit_pass": [None], "sampled_replay": {k: sr_comp.get(k) for k in C2.SR_SUMMARY_KEYS},
            "boundary_linkage": lk_comp, "query_population": qp_comp, "pass": True, "only_arm": "instrumented", "runs": 1}
    summ.update(commit_pass=False)          # `only_arm_summary` over a None row
    C2.seal(out, summ, rows, bundles_of, replay_cfg=REPLAY_CFG, required_manifest=str(man), world=WORLD, versions={"git_head": "c0de"},
             all_pass=True, vc_base=None, comps=comps, xrank_pairs=[{"result": True, "by_family": {}}], xrank_families=[], label="t")
    assert summ["pass"] is None and summ["commit_pass"] is None and len(summ["replay_deferred"]["bundles"]) == WORLD
    (out / "runs.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summ, default=str))
    (out / "versions.json").write_text(json.dumps(VERSIONS))
    (out / "finalize_pair0.json").write_text(json.dumps(fins, default=str))
    (out / "coverage.json").write_text(json.dumps({"verdict": "PASS", "identity_coverage": True, "reason": None, "replay_pending_ranks": [0, 1],
                                                   "value_correspondence": vc}))
    (out / "sampled_replay.json").write_text(json.dumps({"sampled_replay": sr_comp, "boundary_linkage": lk_comp}, default=str))
    return types.SimpleNamespace(out=out, rc=rc, man=man, dd=json.loads((out / C2.DEFERRED).read_text()))


def _replayed(d, rank, sr, lk):
    b = next(x for x in d.dd["bundles"] if x["rank"] == rank)
    doc = {"schema": TPR.RESULT_SCHEMA, "pair": 0, "rank": rank, "bundle": b["path"], "manifest_sha256": b["manifest_sha256"], "where": "cpu", "host": "h",
           "sampled_replay": sr, "boundary_linkage": lk, "wall_s": 2.5, "peak_rss_bytes": 1 << 30, "utc": "u"}
    TPR.result_path(d.out, 0, rank).write_text(json.dumps(doc))


def test_seal_names_every_ranks_bundle_by_its_manifest_and_a_decided_commit_seals_nothing(deferred, tmp_path):
    d = deferred
    assert [(b["pair"], b["rank"]) for b in d.dd["bundles"]] == [(0, 0), (0, 1)] and d.dd["pending_pairs"] == [0]
    for b in d.dd["bundles"]:
        m = RB.open_bundle(b["path"])
        assert m["manifest_sha256"] == b["manifest_sha256"] and m["context"]["tp"] == {"world": WORLD, "rank": b["rank"]}
        assert m["context"]["replay"]["seed"] == SEED and m["context"]["required_manifest_sha256"] == hashlib.sha256(d.man.read_bytes()).hexdigest()
    assert not list(d.out.glob("*.partial"))
    out2 = tmp_path / "decided"
    RB.partial(out2, 0, rank=0).mkdir(parents=True)
    summ = {"pass": False}
    C2.seal(out2, summ, [{"pair": 0, "arm": "instrumented", "commit": {"commit_pass": False, "per_rank": {}}}], {0: {0: {}}}, replay_cfg=REPLAY_CFG,
             required_manifest=None, world=WORLD, versions={}, all_pass=False, vc_base=None, comps={}, xrank_pairs=[], xrank_families=[], label=None)
    assert summ == {"pass": False} and not (out2 / C2.DEFERRED).exists() and not RB.partial(out2, 0, rank=0).exists()


@pytest.mark.parametrize("bad_rank", [None, 1])
def test_the_fold_decides_the_commit_as_the_ranks_processes_would_have(deferred, bad_rank):
    d = deferred
    sr = {r: _sr(r != bad_rank, "complete" if r != bad_rank else "mismatch", rank=r, mismatch=int(r == bad_rank)) for r in range(WORLD)}
    lk = {r: _lk(True, r) for r in range(WORLD)}
    _replayed(d, 0, sr[0], lk[0])
    assert TPR.fold(d.out) is None and json.loads((d.out / "summary.json").read_text())["pass"] is None, "a rank's replay is missing"
    assert TPR.main(TPR.Options(commit_out=str(d.out), checkpoint_dir="/ck", fold_only=True)) == 4
    _replayed(d, 1, sr[1], lk[1])
    want = _in_process([_fin(r, copy.deepcopy(sr[r]), copy.deepcopy(lk[r])) for r in range(WORLD)])
    assert TPR.main(TPR.Options(commit_out=str(d.out), checkpoint_dir="/ck", fold_only=True)) == (0 if bad_rank is None else 3)
    row = json.loads((d.out / "runs.jsonl").read_text().splitlines()[0])["commit"]
    summ = json.loads((d.out / "summary.json").read_text())
    assert (row["commit_pass"], row["commit_fail_reason"]) == (want["row"]["commit_pass"], want["row"]["commit_fail_reason"])
    for r in range(WORLD):
        c, w = row["per_rank"][str(r)], want["row"]["per_rank"][str(r)]
        assert (c["commit_pass"], c["commit_fail_reason"]) == (w["commit_pass"], w["commit_fail_reason"])
        assert {k: c[k] for k in ("sampled_replay", "boundary_linkage", "query_population")} == {k: w[k] for k in ("sampled_replay", "boundary_linkage", "query_population")}
        assert c["replay_deferred"]["replayed"]["where"] == "cpu"
    for k in ("sampled_replay", "boundary_linkage", "query_population"):
        assert summ[k] == want[k], k
    assert summ["pass"] is (bad_rank is None) and summ["commit_pass"] is (bad_rank is None)
    cov = json.loads((d.out / "coverage.json").read_text())
    assert "replay_pending_ranks" not in cov and cov["verdict"] == ("PASS" if bad_rank is None else "FAIL")
    assert cov["value_correspondence"]["c2_of_record"]["result"] is (bad_rank is None)
    if bad_rank is not None:
        assert row["commit_fail_reason"].startswith("rank 1: sampled replay (C2 of record, rank 1): mismatch") and cov["reason"] == row["commit_fail_reason"]
    fin = json.loads((d.out / "finalize_pair0.json").read_text())
    assert [f["sampled_replay"]["sampled_replay"]["grade"] for f in fin] == [sr[0]["grade"], sr[1]["grade"]]
    dd = json.loads((d.out / C2.DEFERRED).read_text())
    assert dd["decided"]["pass"] is (bad_rank is None) and dd["pending_pairs"] == [] and len(dd["replayed"]) == WORLD
    assert TPR.main(TPR.Options(commit_out=str(d.out), checkpoint_dir="/ck")) == (0 if bad_rank is None else 3), "a decided record is never replayed again"


def test_a_replay_of_another_bundle_is_refused(deferred):
    d = deferred
    for r in range(WORLD):
        _replayed(d, r, _sr(rank=r), _lk(True, r))
    doc = json.loads(TPR.result_path(d.out, 0, 1).read_text())
    TPR.result_path(d.out, 0, 1).write_text(json.dumps(dict(doc, manifest_sha256="00" * 32)))
    with pytest.raises(TPR.Refused, match="not " + d.dd["bundles"][1]["manifest_sha256"][:16]):
        TPR.fold(d.out)


# ------------------------------------------------------------------------------------------------ one rank, from its bundle

@pytest.fixture
def stubbed(monkeypatch):
    """The rank's Programs, its checkpoint and its C2 stood in for: what `replay_rank` hands each of them is recorded."""
    seen: dict = {}
    wc = {"equal": True, "opened": 0, "servable": 0, "committed_root": "ab" * 32, "opened_by_source": {}, "unopened_n": 0, "wall_s": 0.0, "why": None}
    monkeypatch.setattr(RR, "load", lambda cfg, rank, world: seen.update(load=(cfg, rank, world)) or types.SimpleNamespace(rank=rank))
    monkeypatch.setattr(WR, "CheckpointIndex", lambda *a, **k: "ck")
    monkeypatch.setattr(CPU, "open_weights", lambda bdir, tree, root, ck, **k: seen.update(open=(root, k)) or ({}, dict(wc)))

    def evaluate(p, cfg, **k):
        seen["evaluate"] = (p, cfg, k)
        return _sr(rank=p.rank), _lk(True, p.rank)

    monkeypatch.setattr(RR, "evaluate", evaluate)
    return seen


def test_a_rank_replays_its_bundle_with_the_commits_seed_against_its_own_shards(deferred, stubbed):
    from verity_vllm.check import oracle_compare as OC

    d = deferred
    o = TPR.Options(commit_out=str(d.out), checkpoint_dir="/ck", ranks="1")
    assert TPR.main(o) == 4, "rank 0 is still to replay"
    cfg, rank, world = stubbed["load"]
    assert (rank, world) == (1, WORLD) and cfg["seed"] == SEED and cfg["programs"] == REPLAY_CFG["programs"]
    assert stubbed["open"] == ("ab" * 32, {"threads": 8, "tp": {"world": WORLD, "rank": 1}})
    _p, _cfg, k = stubbed["evaluate"]
    assert isinstance(k["committed"], OC.OpenedReader) and k["run_root"] == d.rc.run_root and k["bmap"] == _bmap(1, d.rc.run_root.hex())
    assert k["weights_check"]["self_consistent"] is True and k["weights_check"]["pinned"] is None and k["manifest_doc"] == json.loads(d.man.read_text())
    res = json.loads(TPR.result_path(d.out, 0, 1).read_text())
    assert res["manifest_sha256"] == d.dd["bundles"][1]["manifest_sha256"] and res["sampled_replay"]["result"] is True and res["peak_rss_bytes"] > 0
    assert TPR.main(TPR.Options(commit_out=str(d.out), checkpoint_dir="/ck", build_rank=["/here/r0", "/here/r1"])) == 0
    assert stubbed["load"][0]["programs"] == {"0": "/here/r0", "1": "/here/r1"} and stubbed["load"][1] == 0


def test_the_ranks_one_process_replays_share_one_checkpoint_index(deferred, stubbed, monkeypatch):
    built, opened, stub = [], [], CPU.open_weights
    monkeypatch.setattr(WR, "CheckpointIndex", lambda *a, **k: built.append(object()) or built[-1])
    monkeypatch.setattr(CPU, "open_weights", lambda bdir, tree, root, ck, **k: opened.append(ck) or stub(bdir, tree, root, ck, **k))
    assert TPR.main(TPR.Options(commit_out=str(deferred.out), checkpoint_dir="/ck")) == 0
    assert len(built) == 1 and opened == built * WORLD


def test_a_replay_that_cannot_run_is_not_evaluated_and_fails_the_rank(deferred, stubbed, monkeypatch):
    monkeypatch.setattr(RR, "load", lambda *a: (_ for _ in ()).throw(RuntimeError("rank 0: REFUSED (not the Program of record): r0: dddd")))
    assert TPR.main(TPR.Options(commit_out=str(deferred.out), checkpoint_dir="/ck")) == 3
    c = json.loads((deferred.out / "runs.jsonl").read_text().splitlines()[0])["commit"]["per_rank"]["0"]
    assert c["commit_pass"] is False and "not-evaluated" in c["commit_fail_reason"] and "REFUSED (not the Program of record)" in c["commit_fail_reason"]


def _tamper_runs(out, fn):
    rows = [json.loads(ln) for ln in (out / "runs.jsonl").read_text().splitlines()]
    fn(rows[0]["commit"])
    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("tamper,named", [
    (lambda out: (out / "replay_bundle_p0_rank1" / "binding_map.json").write_text("{}"), "sha256 is not the manifest's"),
    (lambda out: _tamper_runs(out, lambda c: c["per_rank"]["1"].update(run_root="11" * 32)), "commitment vector|run root of record"),
    (lambda out: _tamper_runs(out, lambda c: c["per_rank"]["1"]["binding"].update(map_digest="22" * 32)), "binding map"),
    (lambda out: _tamper_runs(out, lambda c: c["per_rank"]["1"].update(commit_pass=True)), "no rank awaiting its deferred replay"),
    (lambda out: _tamper_runs(out, lambda c: c.update(tp_run_root="33" * 32)), "TP run root of record"),
])
def test_a_bundle_that_is_not_the_records_is_refused_before_any_replay(deferred, stubbed, tamper, named):
    tamper(deferred.out)
    assert TPR.main(TPR.Options(commit_out=str(deferred.out), checkpoint_dir="/ck", ranks="1")) == 2
    assert "evaluate" not in stubbed and not TPR.result_path(deferred.out, 0, 1).exists()
    with pytest.raises(TPR.Refused, match=named):
        TPR.replay_rank(TPR.Options(commit_out=str(deferred.out), checkpoint_dir="/ck"), deferred.out, deferred.dd, 0, 1)


def test_a_manifest_that_is_not_the_commits_is_refused(deferred, stubbed, tmp_path):
    other = tmp_path / "other.json"
    other.write_text("{}")
    assert TPR.main(TPR.Options(commit_out=str(deferred.out), checkpoint_dir="/ck", required_manifest=str(other))) == 2 and "evaluate" not in stubbed


def test_a_commit_that_deferred_nothing_is_refused(tmp_path):
    assert TPR.main(TPR.Options(commit_out=str(tmp_path), checkpoint_dir="/ck")) == 2


# ------------------------------------------------------------------------------------------------ the rank's process

def _worker(tmp_path, cfg, bmap):
    from verity_vllm.engine import rank_worker as W

    class Stub:
        _deferred_replay = W.TP2CaptureWorkerExtension._deferred_replay

    s = Stub()
    s._tp2_commit = {"rank": 1, "world": WORLD, "replay": cfg, "bmap": bmap, "com": object()}
    return s


def _program(tmp_path, digest, rank=1):
    d = tmp_path / "build_request_r1"
    d.mkdir(exist_ok=True)
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump({"rows": [], "params": [], "digest": digest}, f)
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": digest}, "LP": 4, "T": 1, "tp": {"world": WORLD, "rank": rank}}))
    return {"programs": {"1": str(d)}, "of_record": {"1": ["c" * 64]},
            "requests": [{"request_id": "r0", "arrive_step": 0, "prompt_len": 4, "max_tokens": 2, "prompt_token_ids": [1, 2, 3, 4]}]}


def test_a_rank_defers_its_replay_into_a_partial_bundle(tmp_path, monkeypatch):
    from verity_vllm.check.replay import driver as SR

    out = tmp_path / "out"
    monkeypatch.setattr(SR, "ProgramIndex", lambda *a, **k: types.SimpleNamespace(rows=[], prefix=""))

    def write_store(out_, pair, com, rc, *, digests, rank):
        p = RB.partial(out_, pair, rank=rank)
        (p / "store").mkdir(parents=True)
        digests["store/store.json"] = "ee" * 32
        return p

    monkeypatch.setattr(RB, "write_store", write_store)
    bmap = {"entries": [], "digest": "m"}
    r = _worker(tmp_path, _program(tmp_path, "c" * 64), bmap)._deferred_replay(types.SimpleNamespace(run_root=b"\x01" * 32), {"out": str(out), "pair": 0})
    sr, b = r["sampled_replay"], r["bundle"]
    assert (sr["grade"], sr["result"], sr["rank"], r["boundary_linkage"]["rank"]) == ("deferred", None, 1, 1) and "replay_bundle_p0_rank1" in sr["why"]
    assert b["partial"] == str(RB.partial(out, 0, rank=1)) and b["hashed"] == {"store/store.json": "ee" * 32} and b["programs"]["r0"]["program_digest"] == "c" * 64
    assert json.loads((RB.partial(out, 0, rank=1) / "binding_map.json").read_text()) == bmap and b["engine_of_record"]


def test_a_rank_whose_program_is_not_of_record_fails_in_place_and_leaves_no_bundle(tmp_path, monkeypatch):
    out = tmp_path / "out"
    RB.partial(out, 0, rank=1).mkdir(parents=True)
    r = _worker(tmp_path, _program(tmp_path, "d" * 64), {"entries": [], "digest": "m"})._deferred_replay(types.SimpleNamespace(run_root=b"\x01" * 32), {"out": str(out), "pair": 0})
    assert r["bundle"] is None and r["sampled_replay"]["grade"] == "not-evaluated" and "REFUSED (not the Program of record)" in r["sampled_replay"]["why"]
    assert not RB.partial(out, 0, rank=1).exists()
    ok, why = C2.c2_and_pin_gates({}, {"sampled_replay": r, "weights_pin": {"result": True}}, True, None, rank=1, replay_on=True, required=True, bundle="b")
    assert ok is False and "not-evaluated" in why


def test_a_rank_without_a_binding_map_defers_nothing(tmp_path):
    r = _worker(tmp_path, {}, None)._deferred_replay(types.SimpleNamespace(run_root=b"\x01" * 32), {"out": str(tmp_path), "pair": 0})
    assert r["sampled_replay"]["grade"] == "not-evaluated" and "bundle" not in r


# ------------------------------------------------------------------------------------------------ the row

def test_a_tp_row_defers_its_replay_and_tp_commit_takes_the_flag(tmp_path):
    from tests.pipeline.test_config_run import _tp
    from verity_vllm.pipeline import cli

    for argv in ((), ("--config-run", "1")):
        t = _tp(tmp_path / str(len(argv)), *argv, "--replay-deferred", "1")
        cmd = t.commit_cmd([])
        assert cli.parse("tp-commit", cmd[cmd.index("tp-commit") + 1:], {}).replay_deferred == "full"
        assert "--replay-deferred" not in _tp(tmp_path / f"plain{len(argv)}", *argv).commit_cmd([])


@pytest.mark.parametrize("world", [None, 8])
def test_a_tp_row_refuses_a_slim_bundle_by_name(tmp_path, world):
    from tests.pipeline.test_row import TP_DENSE, _opts
    from verity_vllm.config import UsageError
    from verity_vllm.pipeline.row import run_row

    o = _opts(tmp_path, "--replay-deferred", "slim", *(("--world", str(world)) if world else ()), row=TP_DENSE)
    with pytest.raises(UsageError, match="slim plan is a single-GPU"):
        run_row(o, o.stages)


@pytest.mark.parametrize("rc_replay", [0, 3])
def test_a_pending_tp_commit_waits_for_the_replay_stage_which_decides_it(tmp_path, monkeypatch, rc_replay):
    from tests.pipeline.test_config_run import _tp
    from verity_vllm.pipeline import row_records as R_
    from verity_vllm.pipeline.row_driver import Stop

    t = _tp(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    (Path(t.d) / "commit").mkdir()
    bundles = [{"pair": 0, "rank": r, "path": f"{t.d}/commit/replay_bundle_p0_rank{r}", "manifest_sha256": "ab" * 32} for r in range(WORLD)]
    (Path(t.d) / "commit" / "summary.json").write_text(json.dumps({"only_arm": "instrumented", "commit_pass": None, "pass": None,
                                                                    "replay_deferred": {"bundles": bundles}}))
    (Path(t.d) / "commit" / C2.DEFERRED).write_text("{}")
    monkeypatch.setattr(R_, "tp_rank_workload_digest", lambda bdir, r: f"wd{r}")
    monkeypatch.setattr(R_, "tp_manifest_complete", lambda d: True)
    monkeypatch.setattr(R_, "applicable_taps", lambda man, ck: [])
    t.hidden_so = str(tmp_path / "verity_fa2_matReqG.so")
    Path(t.hidden_so).touch()
    ran = []
    monkeypatch.setattr(t, "run_tl", lambda stage, to, log, cmd: ran.append((stage, cmd)) or (0 if stage == "commit.tp_commit" else rc_replay))
    t.commit()
    assert t.replay_pending and "commit PENDING" in (Path(t.d) / "stages.txt").read_text()
    assert all(Path(b["path"]).name in (Path(t.d) / "stages.txt").read_text() for b in bundles)
    t.ck = str(tmp_path)
    if rc_replay:
        with pytest.raises(Stop):
            t.replay()
    else:
        t.replay()
    stage, cmd = ran[-1]
    assert stage == "replay.tp_commit_replay" and cmd[cmd.index("tp-commit-replay") + 1] == f"{t.d}/commit" and cmd[cmd.index("--checkpoint-dir") + 1] == t.ck
    assert ("replay PASS" if rc_replay == 0 else "replay FAIL") in (Path(t.d) / "stages.txt").read_text()


class _AtLease(Exception):
    pass


@pytest.mark.parametrize("world, defers", [(2, True), (4, False), (8, True)])
def test_a_tp_commit_at_an_unproven_world_drops_the_deferral_before_it_takes_its_gpus(tmp_path, monkeypatch, capsys, world, defers):
    """The row's `tp-commit --replay-deferred` (REPLAY_DEFERRED=1, as the dispatcher's auto resolves it) run to its GPU lease: at world 4
    the flag is gone by then, so no rank's finalize defers and nothing is sealed, and the Commit's log says why; worlds 2 and 8, where
    deferral is proven, still defer."""
    pytest.importorskip("torch")
    import sys
    from tests.pipeline.test_config_run import _tp
    from verity_vllm.engine import engine_profile, vllm_adapter
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline.tp import lease

    t = _tp(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    t.W = world
    cmd = t.commit_cmd([])
    args = cli.parse("tp-commit", cmd[cmd.index("tp-commit") + 1:], {})
    assert (args.tp, args.replay_deferred) == (world, "full")

    def acquire(*a, **k):
        raise _AtLease(args.replay_deferred)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=None))
    monkeypatch.setattr(engine_profile, "apply_env", lambda *a, **k: None)
    monkeypatch.setattr(vllm_adapter, "load_workload", lambda *a, **k: {})
    monkeypatch.setattr(lease, "acquire", acquire)
    with pytest.raises(_AtLease) as at:
        TC.main(args)
    assert at.value.args[0] == ("full" if defers else None)
    said = [ln for ln in capsys.readouterr().out.splitlines() if "--replay-deferred not taken" in ln]
    assert said == ([] if defers else ["[tp2commit] --replay-deferred not taken (world 4: deferral is proven at TP2 and TP8 only): every "
                                       "rank's C2 replay runs in its process"])


def test_a_tp_commit_that_replayed_in_process_at_an_unproven_world_leaves_the_replay_stage_nothing_to_do(tmp_path, monkeypatch):
    """World 4 under REPLAY_DEFERRED=1: the Commit is decided, not PENDING, and the dispatcher's `row stage replay` runs no CPU replay and
    does not fail (its world-2 control is `test_a_pending_tp_commit_waits_for_the_replay_stage_which_decides_it`)."""
    from tests.pipeline.test_config_run import _tp
    from verity_vllm.pipeline import row_records as R_

    t = _tp(tmp_path, "--config-run", "1", "--replay-deferred", "1")
    t.W = 4
    (Path(t.d) / "commit").mkdir()
    (Path(t.d) / "commit" / "summary.json").write_text(json.dumps({"only_arm": "instrumented", "commit_pass": True, "pass": True,
                                                                    "replay_deferred": {"skipped": "world 4: deferral is proven at TP2 and TP8 only"}}))
    monkeypatch.setattr(R_, "tp_rank_workload_digest", lambda bdir, r: f"wd{r}")
    monkeypatch.setattr(R_, "tp_manifest_complete", lambda d: True)
    monkeypatch.setattr(R_, "applicable_taps", lambda man, ck: [])
    t.hidden_so = str(tmp_path / "verity_fa2_matReqG.so")
    Path(t.hidden_so).touch()
    ran = []
    monkeypatch.setattr(t, "run_tl", lambda stage, to, log, cmd: ran.append(stage) or 0)
    t.commit()
    assert not t.replay_pending and "commit PASS" in (Path(t.d) / "stages.txt").read_text()
    t.replay()
    assert ran == ["commit.tp_commit"] and "replay" not in (Path(t.d) / "stages.txt").read_text()
    assert "nothing deferred, the Commit replayed every rank in process (world 4: deferral" in (Path(t.d) / "row.log").read_text()


def test_a_slim_bundle_at_an_unproven_world_is_not_refused_and_reaches_tp_commit_as_its_flag(tmp_path, monkeypatch):
    from tests.pipeline.test_config_run import _tp
    from tests.pipeline.test_row import TP_DENSE, _opts
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline.row import run_row
    from verity_vllm.pipeline.row_tp import TpRow

    worlds = []
    monkeypatch.setattr(TpRow, "run", lambda self: worlds.append(self.W) or 0)
    o = _opts(tmp_path, "--replay-deferred", "slim", "--world", "4", row=TP_DENSE)
    assert run_row(o, o.stages) == 0 and worlds == [4]
    t = _tp(tmp_path / "t", "--replay-deferred", "slim")
    t.W = 4
    cmd = t.commit_cmd([])
    assert cli.parse("tp-commit", cmd[cmd.index("tp-commit") + 1:], {}).replay_deferred == "full"
