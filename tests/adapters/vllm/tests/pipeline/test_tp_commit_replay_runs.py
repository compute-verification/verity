"""`verity_vllm.pipeline.tp.commit_replay`: a bundle sealed for another run, rank or world than the one it is listed under is refused
before any replay; the replay's own worker count replaces the Commit's; and the fold of a Commit of two instrumented runs leaves the
later run's coverage file as the Commit in one process leaves it: none for a run that passed, its failure for one that did not."""
from __future__ import annotations

import hashlib
import json
import shutil
import types
from pathlib import Path
from typing import Any

import pytest
from verity_vllm.check import weights_of_record as WR
from verity_vllm.commit import step_rows
from verity_vllm.commit import store_dump as SD
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.pipeline.tp import commit_replay as TPR
from verity_vllm.pipeline.tp import rank_replay as RR
from verity_vllm.pipeline.tp import run_root as RT
from verity_vllm.program.frontend import collective_kinds as CK

from tests.check.opened import commit_steps

WORLD = 2


SEED = 7


REPLAY_CFG = {"programs": {"0": "/b/rank0", "1": "/b/rank1"}, "of_record": {"0": ["a" * 64], "1": ["b" * 64]}, "requests": [],
              "per_stratum": 2, "max_evaluate": 2000, "seed": SEED, "workers": None, "challenge": None, "targets": [], "ring_slots": None}


VERSIONS = {"sampled_replay": {"of_record": REPLAY_CFG["of_record"], "seed": SEED}}


XRANK = {"result": True, "by_family": {}}


XF = CK.cross_rank_families(WORLD)


def _sr(result=True, grade="complete", *, rank=0, qp_ok=True, mismatch=0):
    """A rank's sampled-replay record as `driver.sampled_replay` writes it (population reconciled against Q(P) unless qp_ok is not True)."""
    pop = {"vus": 10, "strata_n": 3, "source": "rows", "by_family": {"Gemm_v1": 10}, "query_total": {"identities": 10}}
    if qp_ok is not None:
        outside = [] if qp_ok else [{"request_id": "r0", "step": 0, "op_path": "x", "member": "y"}]
        pop["query_checks"] = {"ok": qp_ok, "vus_outside_query": {"n": 2 * len(outside), "first": outside},
                               **{k: {"n": 0, "first": []} for k in ("identities_without_rows", "rows_disagree")},
                               "ranks": {"replay": [rank], "manifest": [rank], "equal": True}, "accounted": True, "unstated_families": []}
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


def _bmap(rank, root):
    entries = [{"identity": {"rank": rank, "op": "model.embed_tokens"}, "kind": "leaf", "step_root": "00", "leaf_lo": 0, "leaf_hi": 1}]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return {"schema": "m", "run_root": root, "rank": rank, "entries_n": 1, "digest": digest, "entries": entries}


def _cblock(rank, root, bmap):
    return {"run_root": root, "steps": 2, "leaves": 3, "bytes_bound": 516, "tensors": 2, "openings": {"n": 64, "verified": 64},
            "classes": {}, "weight_registration": {"root": "ab" * 32}, "binding": {"map_digest": bmap["digest"], "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}}}


def _seal_deferred(tmp_path):
    """A one-run TP2 Commit (the config run's only-arm form) as `tp-commit --replay-deferred` leaves it: every rank PENDING, its bundle
    sealed (a real dumped store of one committed run, its binding map, its weights tree), `replay_deferred.json` naming them.  Every
    rank commits under rows-sha512 and the TP run root is its v2 form."""
    pytest.importorskip("torch")

    out = tmp_path / "commit"
    out.mkdir()
    com, rc = commit_steps({0: [("model.embed_tokens", bytes(range(256)) * 2)], 1: [("runner.sampler", b"\x01\x02\x03\x04")]})
    root = rc.run_root.hex()
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"identities": [], "ranks": WORLD}))
    bundles_of: dict[int, dict[int, Any]] = {0: {}}
    per_rank, fins = {}, []
    for r in range(WORLD):
        bmap = _bmap(r, root)
        part = RB.partial(out, 0, rank=r)
        SD.dump(com, rc, part / "store")
        (part / "weights").mkdir()
        (part / "weights" / "tree.json").write_text(json.dumps({"names": [], "tensor_roots": [], "root": "ab" * 32, "chunk": step_rows.ROW_BYTES, "extra": {}}))
        (part / "binding_map.json").write_text(json.dumps(bmap))
        f = _deferred_fin(r, out)
        cb = _cblock(r, root, bmap)
        ok, why = C2.c2_and_pin_gates(cb, f, True, None, rank=r, replay_on=True, required=True, bundle=RB.path(out, 0, rank=r).name, xrank_families=XF)
        cb["commit_pass"], cb["commit_fail_reason"] = ok, why
        per_rank[r], bundles_of[0][r] = cb, {"partial": str(part), "hashed": {}, "engine_of_record": {"vllm": "test"}, "programs": {}}
        fins.append({k: v for k, v in f.items()})
    vec = {r: rc.run_root for r in range(WORLD)}
    rows = [{"pair": 0, "arm": "instrumented", "tokens_equal_ref": True, "wall_s": 1.0,
             "commit": {"tp_run_root": RT.combine(RT.TP_ROOT_TAG, WORLD, vec).hex(), "world": WORLD, "per_rank": per_rank,
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
             all_pass=True, vc_base=None, comps=comps, xrank_pairs=[XRANK], xrank_families=sorted(XF), label="t")
    assert summ["pass"] is None and summ["commit_pass"] is None and len(summ["replay_deferred"]["bundles"]) == WORLD
    (out / "runs.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summ, default=str))
    (out / "versions.json").write_text(json.dumps({**VERSIONS, "root_tag": RT.TP_ROOT_TAG.decode()}))
    (out / "finalize_pair0.json").write_text(json.dumps(fins, default=str))
    (out / "coverage.json").write_text(json.dumps({"verdict": "PASS", "identity_coverage": True, "reason": None, "replay_pending_ranks": [0, 1],
                                                   "value_correspondence": vc}))
    (out / "sampled_replay.json").write_text(json.dumps({"sampled_replay": sr_comp, "boundary_linkage": lk_comp}, default=str))
    return types.SimpleNamespace(out=out, rc=rc, man=man, dd=json.loads((out / C2.DEFERRED).read_text()))


@pytest.fixture
def stubbed(monkeypatch):
    """The rank's Programs, its checkpoint and its C2 stood in for: what `replay_rank` hands each of them is recorded."""
    seen: dict = {}
    wc = {"equal": True, "opened": 0, "servable": 0, "committed_root": "ab" * 32, "opened_by_source": {}, "unopened_n": 0, "wall_s": 0.0, "why": None}
    monkeypatch.setattr(RR, "load", lambda cfg, rank, world: seen.update(load=(cfg, rank, world)) or types.SimpleNamespace(rank=rank, loaded={}, index=types.SimpleNamespace(prefix="")))
    monkeypatch.setattr(WR, "CheckpointIndex", lambda *a, **k: "ck")
    monkeypatch.setattr(CPU, "open_weights", lambda bdir, tree, root, ck, **k: seen.update(open=(root, k)) or ({}, dict(wc)))

    def evaluate(p, cfg, **k):
        seen["evaluate"] = (p, cfg, k)
        return _sr(rank=p.rank), _lk(True, p.rank)

    monkeypatch.setattr(RR, "evaluate", evaluate)
    return seen

stubbed = stubbed       # the fixture of the deferred Commit's own tests: the rank's Programs, checkpoint and C2 stood in for
WORLD = WORLD


def _sealed(tmp_path: Path, monkeypatch, **replay: Any) -> types.SimpleNamespace:
    """A sealed deferred Commit of one run (`_seal_deferred`), its replay configuration with ``replay`` over it."""
    monkeypatch.setitem(globals(), "REPLAY_CFG", dict(REPLAY_CFG, **replay))
    return _seal_deferred(tmp_path)


def _options(d: types.SimpleNamespace, **kw: Any) -> TPR.Options:
    return TPR.Options(commit_out=str(d.out), checkpoint_dir="/ck", **kw)


def _record(d: types.SimpleNamespace) -> dict[str, Any]:
    return json.loads((d.out / C2.DEFERRED).read_text())


# ------------------------------------------------------------------------------------------------ a bundle of another run, rank or world

def _another_world(d: types.SimpleNamespace) -> tuple[int, int]:
    (d.out / C2.DEFERRED).write_text(json.dumps(dict(_record(d), world=3)))
    return 0, 1


def _another_rank(d: types.SimpleNamespace) -> tuple[int, int]:
    """Rank 0's sealed bundle at rank 1's path, listed there by its own manifest."""
    dd = _record(d)
    shutil.rmtree(RB.path(d.out, 0, rank=1))
    shutil.copytree(RB.path(d.out, 0, rank=0), RB.path(d.out, 0, rank=1))
    dd["bundles"][1]["manifest_sha256"] = dd["bundles"][0]["manifest_sha256"]
    (d.out / C2.DEFERRED).write_text(json.dumps(dd))
    return 0, 1


def _another_run(d: types.SimpleNamespace) -> tuple[int, int]:
    """Run 0's sealed bundle of rank 1 at run 1's path, listed there by its own manifest."""
    dd = _record(d)
    shutil.copytree(RB.path(d.out, 0, rank=1), RB.path(d.out, 1, rank=1))
    dd["bundles"].append(dict(dd["bundles"][1], pair=1, path=str(RB.path(d.out, 1, rank=1))))
    (d.out / C2.DEFERRED).write_text(json.dumps(dd))
    return 1, 1


@pytest.mark.parametrize("put,context,asked", [
    (_another_world, "pair 0 tp {'world': 2, 'rank': 1}", "pair 0 rank 1 of world 3"),
    (_another_rank, "pair 0 tp {'world': 2, 'rank': 0}", "pair 0 rank 1 of world 2"),
    (_another_run, "pair 0 tp {'world': 2, 'rank': 1}", "pair 1 rank 1 of world 2"),
], ids=["another world", "another rank", "another run"])
def test_a_bundle_sealed_for_another_run_rank_or_world_is_refused_before_any_replay(tmp_path, monkeypatch, stubbed, put, context, asked):
    d = _sealed(tmp_path, monkeypatch)
    pair, rank = put(d)
    with pytest.raises(TPR.Refused) as e:
        TPR.replay_rank(_options(d), d.out, _record(d), pair, rank)
    assert str(e.value) == f"{RB.path(d.out, pair, rank=rank)}: its context is {context}, not {asked}"
    assert stubbed == {} and not TPR.result_path(d.out, pair, rank).exists()
    assert TPR.main(_options(d, ranks="1")) == 2 and not TPR.result_path(d.out, pair, rank).exists()


def test_a_bundle_listed_under_its_own_run_rank_and_world_is_replayed(tmp_path, monkeypatch, stubbed):
    d = _sealed(tmp_path, monkeypatch)
    doc = TPR.replay_rank(_options(d), d.out, _record(d), 0, 1)
    assert (doc["pair"], doc["rank"], doc["bundle"]) == (0, 1, str(RB.path(d.out, 0, rank=1))) and stubbed["load"][1:] == (1, WORLD)


# ------------------------------------------------------------------------------------------------ the replay's worker count

@pytest.mark.parametrize("own,want", [(None, 3), (5, 5), (0, 0)], ids=["none named: the Commit's", "five", "zero is a count named"])
def test_the_replays_own_worker_count_replaces_the_commits(tmp_path, monkeypatch, stubbed, own, want):
    d = _sealed(tmp_path, monkeypatch, workers=3)
    TPR.replay_rank(_options(d, replay_workers=own), d.out, _record(d), 0, 1)
    cfg = stubbed["load"][0]
    assert cfg == dict(REPLAY_CFG, workers=want) and stubbed["evaluate"][1] == cfg


# ------------------------------------------------------------------------------------------------ a Commit of two runs, folded

PENDING = {"verdict": "PASS", "identity_coverage": True, "reason": None, "replay_pending_ranks": [0, 1], "world": WORLD}


def _two_runs(d: types.SimpleNamespace, coverage: dict[str, Any] | None = PENDING) -> None:
    """The record of a deferred Commit of two instrumented runs, made of the one-run Commit's: run 1's row, listed bundles and finalize
    records are run 0's under run 1's number (the fold opens no bundle), and ``coverage`` is the coverage file the Commit wrote for
    run 1 while its ranks were pending (None: no such file)."""
    row = json.loads((d.out / "runs.jsonl").read_text())
    (d.out / "runs.jsonl").write_text("".join(json.dumps(dict(row, pair=p)) + "\n" for p in (0, 1)))
    dd = _record(d)
    dd["bundles"] += [dict(b, pair=1, manifest_sha256=f"{b['rank'] + 1:064x}") for b in dd["bundles"]]
    dd["pending_pairs"], dd["xrank"]["1"] = [0, 1], dd["xrank"]["0"]
    (d.out / C2.DEFERRED).write_text(json.dumps(dd))
    shutil.copy(d.out / "finalize_pair0.json", d.out / "finalize_pair1.json")
    if coverage is not None:
        (d.out / "coverage_pair1.json").write_text(json.dumps(coverage))


def _replay(d: types.SimpleNamespace, *failed: tuple[int, int]) -> None:
    """Every rank's replay of both runs on the record: complete and linked, but each of ``failed``'s (run, rank), which mismatched."""
    for b in _record(d)["bundles"]:
        ok = (b["pair"], b["rank"]) not in failed
        sr = _sr(ok, "complete" if ok else "mismatch", rank=b["rank"], mismatch=int(not ok))
        doc = {"schema": TPR.RESULT_SCHEMA, "pair": b["pair"], "rank": b["rank"], "bundle": b["path"], "manifest_sha256": b["manifest_sha256"], "where": "cpu",
               "host": "h", "sampled_replay": sr, "boundary_linkage": _lk(True, b["rank"]), "wall_s": 2.5, "peak_rss_bytes": 1 << 30, "utc": "u"}
        TPR.result_path(d.out, b["pair"], b["rank"]).write_text(json.dumps(doc))


def _coverage(d: types.SimpleNamespace, pair: int) -> dict[str, Any] | None:
    p = d.out / ("coverage.json" if pair == 0 else f"coverage_pair{pair}.json")
    return {k: v for k, v in json.loads(p.read_text()).items() if k != "value_correspondence"} if p.is_file() else None


def _rows(d: types.SimpleNamespace) -> list[tuple[int, bool, str]]:
    rows = [json.loads(ln) for ln in (d.out / "runs.jsonl").read_text().splitlines()]
    return [(r["pair"], r["commit"]["commit_pass"], r["commit"]["commit_fail_reason"]) for r in rows]


MISMATCH = "rank 1: sampled replay (C2 of record, rank 1): mismatch"
RUN0 = {"verdict": "PASS", "identity_coverage": True, "reason": None}


@pytest.mark.parametrize("coverage", [PENDING, None], ids=["a coverage file of its pending ranks", "no coverage file"])
def test_a_later_run_that_passes_keeps_no_coverage_file(tmp_path, monkeypatch, coverage):
    d = _sealed(tmp_path, monkeypatch)
    _two_runs(d, coverage)
    _replay(d)
    assert TPR.fold(d.out) is True and _rows(d) == [(0, True, "PASS"), (1, True, "PASS")]
    assert (_coverage(d, 0), _coverage(d, 1)) == (RUN0, None)
    summ, dd = json.loads((d.out / "summary.json").read_text()), _record(d)
    assert (summ["pass"], summ["commit_pass"], summ["runs"]) == (True, True, 2)
    assert [(r["pair"], r["rank"]) for r in dd["replayed"]] == [(0, 0), (0, 1), (1, 0), (1, 1)] and dd["pending_pairs"] == [] and dd["decided"]["pass"] is True


def test_a_later_run_that_fails_keeps_its_coverage_file_as_a_failure_with_its_reason(tmp_path, monkeypatch):
    d = _sealed(tmp_path, monkeypatch)
    _two_runs(d)
    _replay(d, (1, 1))
    assert TPR.fold(d.out) is False
    (p0, ok0, why0), (p1, ok1, why1) = _rows(d)
    assert (p0, ok0, why0, p1, ok1) == (0, True, "PASS", 1, False) and why1.startswith(MISMATCH)
    assert _coverage(d, 0) == RUN0, "the first run passed: its coverage stands, with no rank pending"
    assert _coverage(d, 1) == {"verdict": "FAIL", "identity_coverage": False, "reason": why1, "world": WORLD}
    summ = json.loads((d.out / "summary.json").read_text())
    assert (summ["pass"], summ["commit_pass"], summ["runs"]) == (False, False, 2)


def test_a_later_run_two_ranks_of_which_fail_names_each_ranks_reason_in_its_coverage(tmp_path, monkeypatch):
    d = _sealed(tmp_path, monkeypatch)
    _two_runs(d)
    _replay(d, (1, 0), (1, 1))
    assert TPR.fold(d.out) is False
    why = _rows(d)[1][2]
    first, second = why.split("; rank 1: ")
    assert first.startswith("rank 0: sampled replay (C2 of record, rank 0): mismatch") and ("rank 1: " + second).startswith(MISMATCH)
    assert _coverage(d, 1) == {"verdict": "FAIL", "identity_coverage": False, "reason": why, "world": WORLD}


def test_a_first_run_that_fails_fails_its_own_coverage_and_a_later_run_that_passes_keeps_none(tmp_path, monkeypatch):
    d = _sealed(tmp_path, monkeypatch)
    _two_runs(d)
    _replay(d, (0, 1))
    assert TPR.fold(d.out) is False
    (_, ok0, why0), (_, ok1, why1) = _rows(d)
    assert (ok0, ok1, why1) == (False, True, "PASS") and why0.startswith(MISMATCH)
    assert (_coverage(d, 0), _coverage(d, 1)) == ({"verdict": "FAIL", "identity_coverage": False, "reason": why0}, None)


def test_a_later_run_that_fails_with_no_coverage_file_is_given_none(tmp_path, monkeypatch):
    d = _sealed(tmp_path, monkeypatch)
    _two_runs(d, None)
    _replay(d, (1, 1))
    assert TPR.fold(d.out) is False and _coverage(d, 1) is None and _coverage(d, 0) == RUN0


@pytest.mark.parametrize("before,after", [
    ({"verdict": "NOT_RUN", "identity_coverage": None}, {"verdict": "NOT_RUN", "identity_coverage": None}),
    ({"verdict": "PASS", "identity_coverage": None}, {"verdict": "FAIL", "identity_coverage": None}),
    ({"verdict": "NOT_RUN", "identity_coverage": True}, {"verdict": "NOT_RUN", "identity_coverage": False}),
    ({"identity_coverage": True}, {"verdict": "FAIL", "identity_coverage": False}),
], ids=["not run and uncounted", "passed and uncounted", "not run and counted", "no verdict"])
def test_a_failed_runs_coverage_that_was_not_run_stays_not_run_and_one_never_counted_stays_uncounted(tmp_path, monkeypatch, before, after):
    d = _sealed(tmp_path, monkeypatch)
    _two_runs(d, dict(before, replay_pending_ranks=[1]))
    _replay(d, (1, 1))
    assert TPR.fold(d.out) is False
    assert _coverage(d, 1) == dict(after, reason=_rows(d)[1][2])
