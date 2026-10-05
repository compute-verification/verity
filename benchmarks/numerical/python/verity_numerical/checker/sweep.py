"""Run the WP2 reference checker over ``bench-instances/v1``: every positive must be accepted with the recorded word
(and, for ``vu-k1536``, reproduce every recorded intermediate accumulator), every ``wrong`` negative must be rejected
with its claimed word and accepted with the correct one, every ``reject`` negative must be rejected whatever is claimed.

    python -m verity_numerical.checker.sweep [--root DIR] [--tiers vu-k1536,tu-k16,vu-k1536-neg] [--limit N] [--procs P]

The operand arrays are built, not committed (``python -m verity_numerical.bench.instances build``).  Under ``research run``
(``RESEARCH_RUN_DIR`` set) the report goes into the run directory and a ``result.json`` envelope is written beside it;
``validation.status`` is ``passed`` only when every instance behaved as its tier says.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from verity_numerical.bench import instances as I
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.params import REAL, Reject
from verity_numerical.checker.step import (
    accepts_step,
    accepts_vu,
    check_step_expr,
    gen_step,
    gen_vu,
)

__all__ = ["main", "sweep", "sweep_tier"]

_TIER: I.VuTier | I.TuTier | None = None  # set in each worker by the initializer (fork: inherited, no re-read)


def _init(tier: I.VuTier | I.TuTier) -> None:
    global _TIER
    _TIER = tier


def _vu(i: int) -> dict[str, Any]:
    """One ``vu-k1536`` positive: the chain step by step against the recorded accumulators, then the epilogue."""

    t = _TIER
    assert isinstance(t, I.VuTier)
    x, w, y = t.instance(i)
    accs = t.accs(i)
    k = I.STEP
    ctx, c = IntCtx(), 0
    try:
        for s, start in enumerate(range(0, len(x), k)):
            c = check_step_expr(ctx, REAL, c, x[start:start + k], w[start:start + k], gen_step(REAL, c, x[start:start + k], w[start:start + k]))
            if c != accs[s]:
                return {"id": i, "ok": False, "why": f"acc[{s}] checker {c:#010x} != recorded {accs[s]:#010x}"}
        got = Epilogue.check(ctx, REAL, c, Epilogue.gen(REAL, c))
    except Reject as e:
        return {"id": i, "ok": False, "why": f"rejected: {e}"}
    if got != y:
        return {"id": i, "ok": False, "why": f"y checker {got:#06x} != recorded {y:#06x}"}
    return {"id": i, "ok": True}


def _tu(i: int) -> dict[str, Any]:
    """One ``tu-k16`` transition unit with its arbitrary incoming accumulator."""

    t = _TIER
    assert isinstance(t, I.TuTier)
    c, a, b, out = t.instance(i)
    try:
        ok = accepts_step(REAL, c, a, b, out, gen_step(REAL, c, a, b))
    except Reject as e:
        return {"id": i, "ok": False, "why": f"hint generation rejected: {e}"}
    return {"id": i, "ok": ok, **({} if ok else {"why": "recorded word not accepted"})}


def _neg(i: int) -> dict[str, Any]:
    """One negative: ``wrong`` -> claimed rejected and correct accepted; ``reject`` -> claimed rejected."""

    t = _TIER
    assert isinstance(t, I.VuTier)
    x, w, claimed = t.instance(i)
    row = dict(zip(t.columns, t.index[i]))
    try:
        h = gen_vu(REAL, x, w)
    except Reject:
        h = None  # out of the domain at hint generation: rejected
    except Exception as e:  # noqa: BLE001 -- a crash is a finding, not a rejection
        return {"id": i, "ok": False, "mistake": row["mistake"], "why": f"hint generation crashed: {type(e).__name__}: {e}"}
    accepted_claimed = h is not None and accepts_vu(REAL, x, w, claimed, h)
    if accepted_claimed:
        return {"id": i, "ok": False, "mistake": row["mistake"], "why": f"claimed {claimed:#06x} accepted"}
    if row["verdict"] == "wrong" and (h is None or not accepts_vu(REAL, x, w, int(row["correct_y"]), h)):
        return {"id": i, "ok": False, "mistake": row["mistake"], "why": f"correct {int(row['correct_y']):#06x} not accepted"}
    return {"id": i, "ok": True, "mistake": row["mistake"]}


_WORKER = {"vu-k1536": _vu, "tu-k16": _tu, "vu-k1536-neg": _neg}


def sweep_tier(tier_name: str, root: Path, limit: int | None = None, procs: int = 1, log=print) -> dict[str, Any]:
    tier = I.load(tier_name, root)
    n = tier.n if limit is None else min(limit, tier.n)
    fn = _WORKER[tier_name]
    t0 = time.time()
    if procs > 1:
        with Pool(procs, initializer=_init, initargs=(tier,)) as pool:
            results = pool.map(fn, range(n), chunksize=max(1, n // (procs * 8)))
    else:
        _init(tier)
        results = [fn(i) for i in range(n)]
    bad = [r for r in results if not r["ok"]]
    by_mistake: dict[str, dict[str, int]] = {}
    for r in results:
        if "mistake" in r:
            d = by_mistake.setdefault(r["mistake"], {"ok": 0, "bad": 0})
            d["ok" if r["ok"] else "bad"] += 1
    rep = {"tier": tier_name, "checked": n, "of": tier.n, "ok": n - len(bad), "bad": len(bad), "seconds": round(time.time() - t0, 3),
           "failures": bad[:200], **({"by_mistake": by_mistake} if by_mistake else {})}
    log(f"{tier_name}: {rep['ok']}/{n} ok, {len(bad)} bad, {rep['seconds']} s")
    for r in bad[:20]:
        log(f"  {r}")
    return rep


def sweep(root: Path, tiers: list[str], limit: int | None, procs: int, log=print) -> dict[str, Any]:
    doc = {"schema": "checker-sweep/v1", "dataset": I.DATASET,
           "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(), "params": "REAL",
           "tiers": [sweep_tier(t, root, limit, procs, log) for t in tiers]}
    doc["passed"] = all(t["bad"] == 0 and t["checked"] == t["of"] for t in doc["tiers"])
    doc["complete"] = all(t["checked"] == t["of"] for t in doc["tiers"])
    return doc


def _write_result(run_dir: Path, out: Path, doc: dict[str, Any]) -> None:
    status = "passed" if doc["passed"] else ("failed" if doc["complete"] else "not_run")
    detail = "; ".join(f"{t['tier']} {t['ok']}/{t['checked']} ok" + (f" ({t['bad']} bad)" if t["bad"] else "") for t in doc["tiers"])
    env = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"stage": "checker.sweep", "dataset": doc["dataset"], "manifest_sha256": doc["manifest_sha256"],
                                    "params": doc["params"], "tiers": [t["tier"] for t in doc["tiers"]]},
           "validation": {"status": status, "checker": "verity_numerical.checker.sweep", "detail": detail, "evidence": [out.name]},
           "measurements": [m for t in doc["tiers"] for m in (
               {"name": f"{t['tier']}.checked", "value": t["checked"], "unit": "count"},
               {"name": f"{t['tier']}.bad", "value": t["bad"], "unit": "count"},
               {"name": f"{t['tier']}.seconds", "value": t["seconds"], "unit": "s"})],
           "measurement_files": [out.name], "artifacts": [out.name]}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(env, indent=1))
    os.replace(tmp, run_dir / "result.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    ap.add_argument("--root", type=Path, default=I.DEFAULT_ROOT)
    ap.add_argument("--tiers", default=",".join(I.TIERS))
    ap.add_argument("--limit", type=int, default=None, help="first N instances of each tier (smoke)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    tiers = [t for t in a.tiers.split(",") if t]
    bad = sorted(set(tiers) - set(I.TIERS))
    if bad:
        ap.error(f"tiers must be among {I.TIERS}, not {bad}")
    out = a.out or (Path(run_dir) / "checker_sweep.json" if run_dir else Path("backends/numerical/reports/checker_sweep.json"))
    doc = sweep(a.root, tiers, a.limit, a.procs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    any_bad = any(t["bad"] for t in doc["tiers"])
    print(f"{'PASSED' if doc['passed'] else ('FAILED' if any_bad else 'INCOMPLETE (--limit): not the record')}: {out}")
    if run_dir:
        _write_result(Path(run_dir), out, doc)
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
