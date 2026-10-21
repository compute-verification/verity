"""`verity-vllm rejudge RUN`: recompute, with no GPU, every gate of a finished TP Commit that its saved outputs decide.

    verity-vllm rejudge RUN [--commit-dir DIR] [--out FILE]

RUN is the research run of the Commit (`RUN/out/summary.json`, the run's start in `RUN/job.json`) or a directory holding the Commit's
`summary.json`.  Its runs are in the row's `runs.jsonl` (`--commit-dir`, default RUN), which every Commit of the row appends to: this
Commit's are the rows stamped with its `commit_started`, or, for a record before the stamp, the rows written since the run began.

Recomputed, each against what the record says:
  record    the summary is this Commit's: its instrumented runs are in runs.jsonl, in order, under the summary's run roots
  ranks     each instrumented run's verdict from its ranks' (False when one failed, None while one is pending), as `tp.commit` folds it
  manifest  each rank's bound Program digest is the required manifest's (`program_digest_per_rank`)
  xrank     the cross-rank collectives check re-run (`xrank_collectives.check_picks`) over each pair's kept dumps
            (`xrank_pair<k>_rank<r>.pt`), else re-folded (`fold_pairs`) from the per-pair records (`xrank_pair<k>.json`)
  verdict   summary.pass re-assembled from the runs, tokens, value check, fold-Match binding and query population, as `tp.commit` does
`LIVE` names the gates that need the live values, and why.  Exit 0 when every recomputed gate passes, 1 when one is False (each named),
2 when the record cannot be read, and 3 (PENDING) when none is False but a rank's verdict is still pending (a deferred replay not yet run:
`tp-commit-replay`), so a pending Commit reads neither as failed nor as passed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError, option

SCHEMA = "verity-vllm/tp-rejudge/v1"
LIVE = {
    "openings": "each rank's challenge- and population-drawn openings against its run root read the committed bytes",
    "coverage": "identity coverage is computed in the rank against its expected population; binding_map_rank<r>.json keeps pair 0's maps only",
    "match_oracle": "committed TP partials and MoE align witnesses vs the Match capture: the values, and the (step, module) pairing over the "
                    "committer's per-step layouts, which the Commit does not save",
    "attribution": "the committed sampled ids against the Match record",
    "value_check": "tap clones and re-execution of every module output and TP partial (value_check.json keeps their verdicts)",
    "t6_4": "the late-read detector reads the value-check run's clones",
    "sampled_replay": "the C2 replay and boundary linkage read the replay bundles (`tp-commit-replay` re-runs a deferred one on CPU)",
    "weights_pin": "the committed weight shards against the checkpoint's root",
    "tokens": "each run's generated tokens against the first run's",
}


def _short(xs: list) -> list:
    return [(x or "absent")[:16] for x in xs]


def own_runs(rows: list[dict], summ: dict, since: str | None) -> list[dict]:
    """The instrumented runs of the summary's Commit: its stamp, else (a record before the stamp) the runs written since the run began."""
    if summ.get("commit_started"):
        mine = [r for r in rows if r.get("commit_started") == summ["commit_started"]]
    else:
        mine = [r for r in rows if since and str(r.get("utc") or "") >= since]
    return [r for r in mine if r.get("arm") != "control"][:len(summ.get("tp_run_roots") or [])]


def record_gate(rows: list[dict], trt: list[dict], summ: dict, since: str | None) -> dict:
    want, got = summ.get("tp_run_roots") or [], [(r.get("commit") or {}).get("tp_run_root") for r in trt]
    if got == want and want:
        return {"result": True, "runs": [r.get("utc") for r in trt]}
    theirs = sorted({str(r.get("utc")) for r in rows if r.get("arm") != "control" and (r.get("commit") or {}).get("tp_run_root") in want})
    return {"result": False, "why": f"this Commit's runs in runs.jsonl ({'stamp ' + summ['commit_started'] if summ.get('commit_started') else f'since {since}'}) "
                                    f"have roots {_short(got)}, the summary {_short(want)}: the summary is another Commit's"
                                    + (f" (runs with its roots: {theirs[0]} .. {theirs[-1]})" if theirs else "")}


def ranks_gate(trt: list[dict], summ: dict) -> dict:
    per_run, results = [], []
    for i, r in enumerate(trt):
        c = r.get("commit") or {}
        vs = {k: v.get("commit_pass") for k, v in (c.get("per_rank") or {}).items()}
        got = False if False in vs.values() else None if None in vs.values() or not vs else True
        rec = [c.get("commit_pass"), (summ.get("commit_pass") or [None] * len(trt))[i]]
        per_run.append({"pair": r.get("pair"), "recomputed": got, "recorded": rec,
                        "failed": {k: (c["per_rank"][k].get("commit_fail_reason") or "")[:300] for k, v in vs.items() if v is False}})
        results.append(got if rec == [got, got] else False)
    result = None if not trt else False if False in results else None if None in results else True
    return {"result": result, "pending": bool(trt) and result is None, "agrees": all(p["recorded"] == [p["recomputed"]] * 2 for p in per_run),
            "per_run": per_run}


def manifest_gate(trt: list[dict], summ: dict) -> dict:
    want = (summ.get("required_manifest") or {}).get("program_digest_per_rank") or {}
    if not want:
        return {"result": None, "why": "no required manifest"}
    bad = [f"pair {r.get('pair')} rank {k}: bound {_short([got])[0]}, manifest {_short([want.get(k)])[0]}" for r in trt
           for k, c in ((r.get("commit") or {}).get("per_rank") or {}).items() if (got := (c.get("binding") or {}).get("program_digest")) != want.get(k)]
    return {"result": not bad and bool(trt), "mismatches": bad[:20]}


def xrank_gate(cdir: Path, summ: dict, world: int) -> dict:
    from verity_vllm.check.match import xrank_collectives as XC
    files = sorted((p for p in cdir.glob("xrank_pair*.json") if re.fullmatch(r"xrank_pair\d+\.json", p.name)), key=lambda p: int(p.stem[10:]))
    pairs = [json.loads(p.read_text()) for p in files]
    pairs = [p for p in pairs if p.get("commit_started") == summ.get("commit_started")] if summ.get("commit_started") else pairs
    rerun = bool(pairs) and all((cdir / Path(d["file"]).name).exists() for p in pairs for d in p.get("dumps") or [{"file": "-"}])
    if rerun:
        import torch
        pairs = [XC.check_picks({int(d["rank"]): torch.load(cdir / Path(d["file"]).name) for d in p["dumps"]}, world) for p in pairs]
    got, rec = XC.fold_pairs(pairs), summ.get("cross_rank_collectives") or {}
    keys = ("result", "pairs", "picks", "equal", "mismatch_n", "missing_n")
    return {"result": got["result"], "how": "re-run over the kept dumps" if rerun else "re-folded from xrank_pair<k>.json",
            "agrees": all(got.get(k) == rec.get(k) for k in keys), "recomputed": {k: got.get(k) for k in keys}, "recorded": {k: rec.get(k) for k in keys},
            "why": got.get("why")}


def verdict_gate(ranks: dict, summ: dict) -> dict:
    if ranks.get("pending"):
        return {"result": None, "recorded": summ.get("pass"), "why": "a rank's verdict is pending: summary.pass is decided once its deferred replay runs"}
    vc = summ.get("value_check")
    parts = {"runs": ranks["result"] is True, "tokens": all(summ.get("tokens_equal") or []), "value_check": vc is None or vc.get("verdict") == "PASS",
             "fold_match_binding": (summ.get("fold_match_binding") or {}).get("result") is not False,
             "query_population": (summ.get("query_population") or {}).get("result") is not False}
    got = all(parts.values())
    return {"result": got, "agrees": got == summ.get("pass"), "recorded": summ.get("pass"), "parts": parts,
            "not_in_summary_pass": "the cross-rank result (xrank): `tp.commit` folds it into coverage.json's value_correspondence only"}


@dataclass(frozen=True, kw_only=True)
class Options:
    """Recompute a finished TP Commit's GPU-free gates from its saved outputs."""

    run: str = option("run", help="the Commit's research run (RUN/out/summary.json) or a directory with its summary.json")
    commit_dir: str | None = option("--commit-dir", default=None, help="the row's commit/ directory: runs.jsonl, xrank_pair<k>.json (default: RUN)")
    out: str | None = option("--out", default=None, help="write the rejudgement here (JSON)")


def main(a: Options) -> int:
    run = Path(a.run)
    sp = next((p for p in (run / "out" / "summary.json", run / "summary.json") if p.exists()), None)
    cdir = Path(a.commit_dir) if a.commit_dir else run
    if sp is None or not (cdir / "runs.jsonl").exists():
        raise UsageError(f"no summary.json under {run} or no runs.jsonl in {cdir}")
    summ = json.loads(sp.read_text())
    job = run / "job.json"
    since = (json.loads(job.read_text()).get("written_utc") or "")[:19] + "Z" if job.exists() else None
    rows = [json.loads(ln) for ln in (cdir / "runs.jsonl").read_text().splitlines() if ln.strip()]
    trt = own_runs(rows, summ, since)
    gates: dict[str, dict] = {"record": record_gate(rows, trt, summ, since)}
    if gates["record"]["result"]:
        gates["ranks"] = ranks_gate(trt, summ)
        gates["manifest"] = manifest_gate(trt, summ)
        gates["xrank"] = xrank_gate(cdir, summ, int(summ.get("tp") or 1))
        gates["verdict"] = verdict_gate(gates["ranks"], summ)
    failed = [g for g, v in gates.items() if v["result"] is False or v.get("agrees") is False]
    pending = not failed and bool(gates.get("ranks", {}).get("pending"))
    rep = {"schema": SCHEMA, "run": str(run), "summary": str(sp), "commit_dir": str(cdir), "commit_started": summ.get("commit_started"), "since": since,
           "verdict": "FAIL" if failed else "PENDING" if pending else "PASS", "failed": failed, "gates": gates, "live_only": LIVE}
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for g, v in gates.items():
        print(f"[rejudge] {g}: {v['result']}" + ("" if v.get("agrees", True) else " (the record says otherwise)")
              + (f" -- {v['why']}" if v.get("why") and v["result"] is not True else ""), flush=True)
    print(f"[rejudge] {rep['verdict']}" + (f": {', '.join(failed)}" if failed else "") + f"; needs live values: {', '.join(LIVE)}", flush=True)
    return 1 if failed else 3 if pending else 0
