"""`verity-vllm rejudge`: a finished TP Commit's GPU-free gates recomputed from its saved outputs -- its own runs found in the row's
shared runs.jsonl by its stamp (or, before the stamp, since its run began), each run's verdict from its ranks', the bound Programs
against the manifest, the cross-rank check re-run over kept dumps or re-folded, and summary.pass re-assembled; a summary that is not the
run's Commit's is named."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from verity_vllm.check.match import xrank_collectives as XC
from verity_vllm.pipeline.cli import main as cli

STAMP = "2026-10-16T21:55:00Z"
PROG = {"0": "p0" * 64, "1": "p1" * 64}


def _rank(ok: bool | None = True, prog: str | None = None, rank: str = "0") -> dict:
    return {"commit_pass": ok, "commit_fail_reason": "PASS" if ok else "match oracle: oracle step 0: module(s) ['model.layers.1'] recorded more than once",
            "binding": {"program_digest": prog or PROG[rank]}}


def _xpair(pair: int, ok: bool = True, stamp: str | None = STAMP) -> dict:
    return {"schema": "verity/tp2/xrank-collectives/v1", "tp": 2, **({"commit_started": stamp} if stamp else {}), "pair": pair, "result": ok,
            "picks": 4, "equal": 4 if ok else 3, "mismatch_n": 0 if ok else 1, "missing_n": 0, "by_family": {"AllReduce_v1": {"picks": 4, "equal": 4 if ok else 3}},
            "dumps": [{"rank": r, "file": f"/elsewhere/xrank_pair{pair}_rank{r}.pt"} for r in (0, 1)]}


def _commit(d: Path, ranks: list[dict] | None = None, stamp: str | None = STAMP, pairs: int = 2, utc: str = "2026-10-16T22:00:00Z",
            xrank_ok: bool = True) -> Path:
    """A TP2 Commit's record as `tp.commit` writes it: rows appended to runs.jsonl, per-pair cross-rank records, the summary."""
    d.mkdir(parents=True, exist_ok=True)
    per_rank = ranks or [_rank(rank="0"), _rank(rank="1")]
    vs = [r["commit_pass"] for r in per_rank]
    run_pass = False if False in vs else None if None in vs else True
    rows = []
    for p in range(pairs):
        st = {"commit_started": stamp} if stamp else {}
        rows += [{"pair": p, "arm": "control", "utc": utc, **st},
                 {"pair": p, "arm": "instrumented", "utc": utc, **st,
                  "commit": {"tp_run_root": "ab" * 32, "commit_pass": run_pass, "per_rank": {str(i): r for i, r in enumerate(per_rank)}}}]
        (d / f"xrank_pair{p}.json").write_text(json.dumps(_xpair(p, xrank_ok, stamp)))
    with (d / "runs.jsonl").open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    xr = XC.fold_pairs([_xpair(p, xrank_ok, stamp) for p in range(pairs)])
    summ = {**({"commit_started": stamp} if stamp else {}), "tp": 2, "tp_run_roots": ["ab" * 32] * pairs, "commit_pass": [run_pass] * pairs,
            "tokens_equal": [True] * 2 * pairs, "value_check": {"verdict": "PASS"}, "fold_match_binding": {"result": True},
            "query_population": {"result": True}, "cross_rank_collectives": xr, "required_manifest": {"program_digest_per_rank": PROG},
            "pass": run_pass}
    (d / "summary.json").write_text(json.dumps(summ))
    return d


def _rejudge(d: Path, *extra: str) -> tuple[int, dict]:
    rc = cli(["rejudge", str(d), "--out", str(d / "rejudge.json"), *extra])
    return rc, json.loads((d / "rejudge.json").read_text())


def test_a_passing_commit_rejudges_pass_on_every_recomputed_gate(tmp_path: Path) -> None:
    rc, rep = _rejudge(_commit(tmp_path / "c"))
    assert rc == 0 and rep["verdict"] == "PASS" and set(rep["gates"]) == {"record", "ranks", "manifest", "xrank", "verdict"}
    assert all(g["result"] is True and g.get("agrees", True) for g in rep["gates"].values())
    assert rep["gates"]["xrank"]["how"] == "re-folded from xrank_pair<k>.json" and "match_oracle" in rep["live_only"]


def test_a_failed_rank_fails_its_run_and_the_verdict_by_name(tmp_path: Path) -> None:
    rc, rep = _rejudge(_commit(tmp_path / "c", ranks=[_rank(rank="0"), _rank(False, rank="1")]))
    assert rc == 1 and rep["failed"] == ["ranks", "verdict"]
    run0 = rep["gates"]["ranks"]["per_run"][0]
    assert run0["recomputed"] is False and run0["recorded"] == [False, False] and "recorded more than once" in run0["failed"]["1"]
    assert rep["gates"]["verdict"]["agrees"] and rep["gates"]["verdict"]["recorded"] is False


def test_a_pending_rank_rejudges_pending_not_failed(tmp_path: Path) -> None:
    rc, rep = _rejudge(_commit(tmp_path / "c", ranks=[_rank(rank="0"), _rank(None, rank="1")]))
    assert rc == 3 and rep["verdict"] == "PENDING" and rep["failed"] == []
    assert rep["gates"]["ranks"]["result"] is None and rep["gates"]["ranks"]["pending"] and rep["gates"]["ranks"]["agrees"]
    assert rep["gates"]["verdict"]["result"] is None and "deferred replay" in rep["gates"]["verdict"]["why"]


def test_a_failed_rank_beside_a_pending_one_still_fails(tmp_path: Path) -> None:
    rc, rep = _rejudge(_commit(tmp_path / "c", ranks=[_rank(False, rank="0"), _rank(None, rank="1")]))
    assert rc == 1 and rep["verdict"] == "FAIL" and rep["failed"] == ["ranks", "verdict"]


def test_a_rank_bound_to_another_program_than_the_manifest_s_fails_the_manifest_gate(tmp_path: Path) -> None:
    rc, rep = _rejudge(_commit(tmp_path / "c", ranks=[_rank(rank="0"), _rank(prog="ff" * 64, rank="1")]))
    assert rc == 1 and rep["failed"] == ["manifest"] and rep["gates"]["manifest"]["mismatches"][0].startswith("pair 0 rank 1: bound ffff")


def test_the_cross_rank_check_reruns_over_kept_dumps(tmp_path: Path, monkeypatch) -> None:
    d = _commit(tmp_path / "c", xrank_ok=True)
    for p in (0, 1):
        for r in (0, 1):
            torch.save({"rank": r, "picks": []}, d / f"xrank_pair{p}_rank{r}.pt")
    seen = []
    monkeypatch.setattr(XC, "check_picks", lambda dumps, world: seen.append((sorted(dumps), world)) or _xpair(len(seen) - 1, ok=False))
    rc, rep = _rejudge(d)
    assert seen == [([0, 1], 2), ([0, 1], 2)], "each pair's rank dumps, by rank"
    x = rep["gates"]["xrank"]
    assert rc == 1 and x["how"] == "re-run over the kept dumps" and x["result"] is False and not x["agrees"] and rep["failed"] == ["xrank"]


def test_a_summary_that_is_another_commit_s_is_named_and_nothing_else_is_judged(tmp_path: Path) -> None:
    d = _commit(tmp_path / "c", stamp="2026-10-16T13:00:00Z", utc="2026-10-16T13:05:00Z")
    old = (d / "summary.json").read_text()
    _commit(d, stamp=STAMP)
    (d / "summary.json").write_text(old.replace("ab" * 32, "cd" * 32))
    rc, rep = _rejudge(d)
    assert rc == 1 and list(rep["gates"]) == ["record"] and "the summary is another Commit's" in rep["gates"]["record"]["why"]


def test_a_record_before_the_stamp_takes_the_runs_written_since_its_run_began(tmp_path: Path) -> None:
    run = tmp_path / "r"
    _commit(run / "commit", stamp=None, utc="2026-10-16T13:05:00Z", ranks=[_rank(False, rank="0"), _rank(rank="1")])
    _commit(run / "commit", stamp=None, utc="2026-10-16T16:00:00Z")
    (run / "out").mkdir()
    (run / "out" / "summary.json").write_text((run / "commit" / "summary.json").read_text())
    (run / "job.json").write_text(json.dumps({"written_utc": "2026-10-16T15:13:26.637Z"}))
    rc, rep = _rejudge(run, "--commit-dir", str(run / "commit"))
    assert rc == 0 and rep["since"] == "2026-10-16T15:13:26Z" and rep["gates"]["record"]["runs"] == ["2026-10-16T16:00:00Z"] * 2
