"""Re-baseline the reference regression on the integration's (v2) outputs -- and keep the historical comparison runnable.

    cd integrations/vllm
    python -m tests.regression.rebaseline run   --record DIR [--oracle expected|frozen] [--tier T0,T1,T2] [-- <pytest args>]
    python -m tests.regression.rebaseline table --record DIR                      # row x check -> PASS (oracle) | PASS by decision | FAIL
    python -m tests.regression.rebaseline write --record DIR --commit SHA --branch B --release R [--run-id ID ...] [--force] [--dry-run]

`run` is `pytest tests/regression -m regression` with `VERITY_REGRESSION=1`, `VERITY_REGRESSION_RECORD_ACTUAL=DIR` (every check's
`{expected, actual, problems, matched}` lands in `DIR/<row>/<check>.json`, `resolver.record_result`) and `VERITY_REGRESSION_ORACLE`:

    expected  (default)  the contract is `expected/<row>.json` -- after `write`, v2's own outputs; a difference is a regression of v2
    frozen               the contract is lifted from the R2 frozen v1 records (`check.expected(row)`); the labelled decisions of
                         fixtures.toml sanction the known differences -- the historical "does v2 reproduce v1" comparison

`write` merges the recorded results into `expected/<row>.json`: for every check with a recorded, passing result the contract's VALUES
become the actual's (the contract's SHAPE is kept: the keys of the previous expected plus the keys every labelled decision on that row /
check names -- nothing the actual states beyond the contract is added, nothing typed by hand); `reference` becomes the v2 tree the
results were produced by (the previous, frozen v1 reference is kept under `reference.frozen`); `checks.<name>.rebaselined` records
which side the result matched (`decision <finding>` = the reason the new expected differs from the R2 record).  Checks without a
recorded result keep their previous contract and are listed.  A recorded FAILURE refuses the write (`--force` overrides, per check,
and is itself recorded).  Output serialisation is `lift_expected.dumps_expected` (compact per-check lines: the 5 MiB blob cap).
The four accepted decisions stay in fixtures.toml as history; `--oracle frozen` still consults them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import checks, resolver
from .lift_expected import dumps_expected

VOLATILE = ("source", "sanctioned_by", "decision_status")      # bookkeeping `actual` carries, never part of a contract
BLOB_LIMIT = 5 * 1024 * 1024


# ---- run ---------------------------------------------------------------------------------------------------------------------------------

def run(a, extra: list[str]) -> int:
    env = dict(os.environ)
    env["VERITY_REGRESSION"] = "1"
    env[resolver.RECORD_ACTUAL_ENV] = str(Path(a.record).resolve())
    env[resolver.ORACLE_ENV] = a.oracle
    if a.tier:
        env["VERITY_REGRESSION_TIERS"] = a.tier
    Path(a.record).mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "pytest", str(resolver.HERE), "-m", "regression", "-q", "-p", "no:cacheprovider",
            "--junit-xml", str(Path(a.record) / "junit.xml"), *extra]
    print("$ " + " ".join(argv), flush=True)
    return subprocess.call(argv, cwd=str(resolver.VLLM_ROOT), env=env)


# ---- the recorded results ----------------------------------------------------------------------------------------------------------------

def results(record: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for rd in sorted(p for p in record.iterdir() if p.is_dir()):
        for f in sorted(rd.glob("*.json")):
            try:
                doc = json.load(open(f))
            except Exception as e:
                print(f"[rebaseline] unreadable {f}: {e}", file=sys.stderr)
                continue
            if doc.get("schema") != "verity-vllm/regression-result/v1":
                continue
            out.setdefault(rd.name, {})[f.stem] = doc
    return out


def table(a) -> int:
    res = results(Path(a.record))
    rows_ = {r.key: r for r in resolver.rows()}
    keys = sorted(res, key=lambda k: rows_[k].number if k in rows_ else 10**6)
    names = [n for n in checks.ORDER if any(n in res[k] for k in keys)]
    w = max([len("row")] + [len(f"#{rows_[k].number} {k}") for k in keys if k in rows_]) if keys else 3
    print(f"{'row':<{w}}  " + "  ".join(f"{n[:16]:<16}" for n in names))
    for k in keys:
        cells = []
        for n in names:
            d = res[k].get(n)
            if d is None:
                cells.append("-")
            elif d["problems"]:
                cells.append("FAIL")
            elif d["matched"].startswith("decision"):
                cells.append("PASS/decision")
            else:
                cells.append(f"PASS/{d['matched']}")
        label = f"#{rows_[k].number} {k}" if k in rows_ else k
        print(f"{label:<{w}}  " + "  ".join(f"{c:<16}" for c in cells))
    fails = [(k, n) for k in keys for n, d in res[k].items() if d["problems"]]
    if fails:
        print("\nFAIL:")
        for k, n in fails:
            print(f"  {k} {n}:\n    " + "\n    ".join(res[k][n]["problems"][:20]))
    return 1 if fails else 0


# ---- write ---------------------------------------------------------------------------------------------------------------------------------

def contract_keys(row: resolver.Row, check: str, previous: dict | None, exp_used: Any) -> list[str]:
    """The keys the re-baselined contract states: the previous expected's (or, when expected/ never froze this check, the lifted
    contract the run used) plus every key a labelled decision on this row / check names."""
    base = previous if isinstance(previous, dict) else (exp_used if isinstance(exp_used, dict) else {})
    keys = list(base)
    for d in row.decisions(check):
        for k in (d.get("expected") or {}):
            if k not in keys:
                keys.append(k)
    return keys


def merge(row: resolver.Row, old: dict, res: dict[str, dict], *, reference: dict, force: bool) -> tuple[dict, list[str]]:
    doc: dict[str, Any] = dict(old) if old else {"row": row.number, "class": row.cls, "checks": {}}
    doc.setdefault("checks", {})
    notes: list[str] = []
    for name in checks.ORDER:
        chk = checks.load(name)
        r = res.get(name)
        if r is None:
            notes.append(f"{name}: no recorded result (contract unchanged)")
            continue
        if r["problems"] and not force:
            raise SystemExit(f"[rebaseline] {row.key} {name} FAILED in the recorded run; refusing to re-baseline on a failure "
                             f"(--force to override):\n  " + "\n  ".join(r["problems"][:20]))
        act = r["actual"]
        if not isinstance(act, dict):
            notes.append(f"{name}: actual is not an object; skipped")
            continue
        keys = contract_keys(row, name, old.get(name) if old else None, r.get("expected"))
        new = {k: act[k] for k in keys if k in act and k not in VOLATILE}
        dropped = [k for k in keys if k not in act and k not in VOLATILE]
        if dropped:
            notes.append(f"{name}: the actual does not state {dropped} (was in the contract; dropped)")
        doc[name] = new
        meta = {"applies": True, "tier": chk.TIER, "strength": chk.STRENGTH,
                "rebaselined": {"matched": r["matched"], "engine": r.get("engine"), "oracle": r.get("oracle")}}
        if r["problems"]:
            meta["rebaselined"]["forced_over"] = r["problems"][:10]
        doc["checks"][name] = meta
    prev = (old or {}).get("reference") or {}
    ref = dict(reference)
    if prev.get("frozen"):                                   # already re-baselined once: the R2 (v1) reference stays the history
        ref["frozen"] = prev["frozen"]
    elif prev and prev.get("tag") != reference.get("tag"):
        ref["frozen"] = prev                                 # the v1-reference block lift_expected.py wrote
    doc["reference"] = ref
    doc["row"] = row.number
    doc["class"] = row.cls
    return doc, notes


def write(a) -> int:
    record = Path(a.record)
    res = results(record)
    if not res:
        print(f"[rebaseline] no recorded results under {record}", file=sys.stderr)
        return 2
    reference = {"tag": "v2-reference", "release": a.release, "commit": a.commit, "branch": a.branch}
    if a.run_id:
        reference["runs"] = list(a.run_id)
    rc = 0
    for row in resolver.rows():
        if row.key not in res:
            print(f"[rebaseline] #{row.number} {row.key}: no results; expected/ unchanged")
            continue
        old = row.expected()
        doc, notes = merge(row, old, res[row.key], reference=reference, force=a.force)
        text = dumps_expected(doc)
        if len(text.encode()) > BLOB_LIMIT:
            print(f"[rebaseline] #{row.number}: {len(text)} bytes exceeds the {BLOB_LIMIT} blob cap; not written", file=sys.stderr)
            rc = 3
            continue
        p = resolver.EXPECTED_DIR / f"{row.key}.json"
        changed = sorted(n for n in checks.ORDER if n in doc and json.dumps(old.get(n), sort_keys=True) != json.dumps(doc.get(n), sort_keys=True))
        print(f"[rebaseline] #{row.number} {row.key}: rebaselined {sorted(res[row.key])}; contract values changed in {changed or 'none'}"
              + (("; " + "; ".join(notes)) if notes else ""))
        if not a.dry_run:
            resolver.EXPECTED_DIR.mkdir(exist_ok=True)
            p.write_text(text)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--record", required=True)
    r.add_argument("--oracle", choices=resolver.ORACLES, default="expected")
    r.add_argument("--tier", default=None, help="VERITY_REGRESSION_TIERS, e.g. T0,T1,T2")
    t = sub.add_parser("table")
    t.add_argument("--record", required=True)
    w = sub.add_parser("write")
    w.add_argument("--record", required=True)
    w.add_argument("--commit", required=True, help="the tree sha the recorded results were produced by")
    w.add_argument("--branch", required=True)
    w.add_argument("--release", required=True)
    w.add_argument("--run-id", action="append", default=[], help="research run / attempt ids of the recorded runs")
    w.add_argument("--force", action="store_true")
    w.add_argument("--dry-run", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a, extra)
    if a.cmd == "table":
        return table(a)
    return write(a)


if __name__ == "__main__":
    sys.exit(main())
