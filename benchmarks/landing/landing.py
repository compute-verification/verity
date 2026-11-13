"""Mark to `main` (README.md): for each PR the verified queue landed on `main` in a window, the hours from its ready mark at
the head that landed to the move of `main` that holds it, split as the queue splits its batches (plain, Lean), with each
class's p90, median, count and share within 2 h.

    landing.py --repo DIR --out DIR [--days N] [--until TIME]

The window is the `--days` (7) before `--until` (a UTC time, default now). `run.sh` runs it as a research run.

**What is read**, each through the queue's own code:

- landings: the loop's decision log on the control pod (`queue_table.CONTROL_SCRIPT`, over `research pods ssh`): each `advance`
  logged done and not the shadow's whose time is in the window, with its members, their heads and the `main` it moved from.
  A PR is counted once, at its first landing at a head.
- marks: the `ready` labels on `pr:N@HEAD` in the evidence store, pulled from its remote, at the head that landed and at each
  head a restack across a move made it from (`queue_table.restacks`, from GitHub): the mark is the one in force when `main`
  moved (`queue_table.mark_at` over `queue_table.carried`), as `research queue stats` counts mark to `main`.
- the split: the loop's own rule for a member of a Lean batch (`vqueue.Ready.lean`, `vqueue_loop.World._judge`): the head's
  change against its merge base with the `main` it landed on is a Lean change (`queue.lean_change`) that touches a Lean
  source (`queue.LEAN_SOURCES`), not a lock alone. Every other landed PR is plain, a change to `security/`'s proofs alone
  among them.

**Out of the count**: a move of `main` the loop didn't log as an advance (by hand, or drive's before the loop moved `main`),
and a landed PR with no mark in force at its landing, which `landing.json` lists.

**The guard**: every input read (the decision log, every label pull, every head's restack lookup and split). Where it does
not hold, no measurement is written, `landing.json` says what was not read, and the run's validation fails."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from research import queue_table as Q
from verity_bench import kit

DOC, SCHEMA = "landing.json", "landing/v1"
DAYS = 7
#: the queue's target: a PR on `main` within two hours of its mark
TARGET_H = Q.TARGET_MIN / 60
CLASSES = ("plain", "lean")
#: each class's measurements, by name: the first is its metric's id (benchmarks/landing/metrics.toml)
NAMES = {"plain": ("ci/plain-mark-to-main-p90", "ci/plain-mark-to-main-median", "ci/plain-landed", "ci/plain-within-2h"),
         "lean": ("ci/lean-mark-to-main-p90", "ci/lean-mark-to-main-median", "ci/lean-landed", "ci/lean-within-2h")}
WHAT = {"plain": "with no Lean change", "lean": "changing Lean"}


class Unread(Exception):
    """An input that could not be read; its text says which and why."""


# ---- pure ------------------------------------------------------------------------------------------------------------------

def landings(decisions: Iterable[dict], start: float, end: float) -> list[dict]:
    """Each PR the loop landed in [start, end], at its first landing at a head: its number, the head, the `main` its advance
    moved from (`base`), the commit `main` moved to and when, oldest first."""
    seen: dict[tuple[str, str], dict] = {}
    for r in decisions:
        if r.get("action") != "advance" or not r.get("done") or r.get("shadow"):
            continue
        at = Q.ts(r["at"])
        if not start <= at <= end:
            continue
        for pr, head in zip(map(str, r.get("members") or []), r.get("heads") or []):
            if (pr, head) not in seen or at < seen[(pr, head)]["landed"]:
                seen[(pr, head)] = {"pr": pr, "head": head, "base": r.get("base") or "", "sha": r.get("sha") or "", "landed": at}
    return sorted(seen.values(), key=lambda x: (x["landed"], int(x["pr"])))


def rows(landed: list[dict], labels: dict[str, list[dict]], restacks: dict[str, str], lean: dict[str, bool]) -> list[dict]:
    """Each landing with its mark (None where none was in force when `main` moved), its hours from the mark, and its class."""
    out = []
    for x in landed:
        mark = Q.mark_at(Q.carried(labels, restacks, x["pr"], x["head"]), x["landed"])
        out.append({**x, "mark": mark, "hours": None if mark is None else (x["landed"] - mark) / 3600,
                    "class": "lean" if lean[x["head"]] else "plain"})
    return out


def summary(rs: list[dict]) -> dict:
    """Each class's count, median and p90 hours (`queue_table.pct`) and share within `TARGET_H`, and the landings with no mark."""
    out: dict = {"unmarked": sum(r["hours"] is None for r in rs)}
    for c in CLASSES:
        hs = [r["hours"] for r in rs if r["class"] == c and r["hours"] is not None]
        out[c] = {"n": len(hs), "median_h": Q.pct(hs, .5), "p90_h": Q.pct(hs, .9),
                  "within_2h": sum(h <= TARGET_H for h in hs) / len(hs) if hs else None}
    return out


def measurements(s: dict) -> list[dict]:
    """The result's measurements: for each class its p90 (with the target), median, count and share within 2 h; a class with
    no landing has no p90, median or share, and says why."""
    out = []
    for c in CLASSES:
        p90, median, count, share = NAMES[c]
        v = s[c]
        why = None if v["n"] else f"no PR {WHAT[c]} landed in the window"
        out += [{"name": p90, "value": _r(v["p90_h"]), "unit": "hours", "target": TARGET_H, "why": why,
                 "verdict": None if why else "within target" if v["p90_h"] <= TARGET_H else "over target"},
                {"name": median, "value": _r(v["median_h"]), "unit": "hours", "why": why},
                {"name": count, "value": v["n"], "unit": "PRs"},
                {"name": share, "value": _r(v["within_2h"]), "unit": "share of the PRs landed", "why": why}]
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- reading ---------------------------------------------------------------------------------------------------------------

def read_decisions(start: float) -> list[dict]:
    """The loop's decision log from ``start`` on, as `research queue stats` reads it."""
    got, why = Q._remote(Q.CONTROL, Q.CONTROL_SCRIPT, Q.LIVE, iso(start))
    if got is None:
        raise Unread(f"the loop's decision log on {Q.CONTROL}: {why}")
    return got.get("decisions") or []


def read_labels(pairs: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """{`PR@HEAD`: its `ready` labels} for each (PR, head), pulled from the store's remote first; a pull that fails is unread."""
    from research.queue import label_target
    from research.store import labels_sync
    from research.store.local import LocalStore

    store, targets = LocalStore(None), {f"{pr}@{h}": label_target(int(pr), h) for pr, h in pairs}
    try:
        got = labels_sync.pull_targets(store, list(targets.values()), jobs=16)
    except Exception as e:  # noqa: BLE001  any failure of the remote leaves the local copies, which may lack marks
        raise Unread(f"the store's labels: {type(e).__name__}: {e}") from None
    if got.get("bad"):
        raise Unread(f"the store's labels: {len(got['bad'])} that don't validate, first {got['bad'][0]}")
    return {k: [{"key": lab.key, "value": lab.value, "by": lab.by, "ts": lab.ts} for lab in store.labels(t) if lab.key == "ready"]
            for k, t in targets.items()}


def lean_batch(repo: Path, base: str, head: str) -> bool:
    """Whether the loop batches ``head`` landed on ``base`` as a Lean change (`vqueue_loop.World._judge`)."""
    from research.queue import LEAN_SOURCES, lean_change, own_files

    merge_base = subprocess.run(["git", "-C", str(repo), "merge-base", base, head], capture_output=True, text=True, check=True)
    return (lean_change(repo, head, merge_base.stdout.strip())
            and any(f.endswith(LEAN_SOURCES) for f in own_files(repo, base, head)))


def collect(repo: Path, start: float, end: float, decisions: Callable[[float], list[dict]] = read_decisions,
            restacks: Callable[[list[str]], tuple[dict[str, str], list[str]]] = Q.restacks,
            labels: Callable[[list[tuple[str, str]]], dict[str, list[dict]]] = read_labels,
            lean: Callable[[Path, str, str], bool] = lean_batch) -> dict:
    """The document: the window, each landing's row, the summary and what was not read (`errors`)."""
    doc: dict = {"schema": SCHEMA, "start": iso(start), "end": iso(end), "rows": [], "errors": []}
    try:
        landed = landings(decisions(start), start, end)
    except Unread as e:
        doc["errors"].append(str(e))
        return doc
    found, unread = restacks(sorted({x["head"] for x in landed}))
    doc["errors"] += unread
    pairs = sorted({(x["pr"], h) for x in landed for h in Q.chain(found, x["head"])})
    try:
        labs = labels(pairs)
    except Unread as e:
        doc["errors"].append(str(e))
        return doc
    split = {}
    for x in landed:
        try:
            split[x["head"]] = lean(repo, x["base"], x["head"])
        except Exception as e:  # noqa: BLE001  a head git can't read is unread, and the guard fails on it
            doc["errors"].append(f"the split of #{x['pr']} at {x['head'][:12]} on {x['base'][:12]}: {type(e).__name__}: {e}"[:300])
    if doc["errors"]:
        return doc
    rs = rows(landed, labs, found, split)
    doc["rows"] = [{**r, "landed": iso(r["landed"]), "mark": None if r["mark"] is None else iso(r["mark"])} for r in rs]
    doc["summary"] = summary(rs)
    return doc


# ---- writing ---------------------------------------------------------------------------------------------------------------

def write(w: kit.Writer, doc: dict, out: Path, days: float) -> bool:
    """``doc`` in ``out``, named in the result, the result's fingerprint and validation, and where the guard holds its
    measurements. Returns whether it holds."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / DOC
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    rel = path.relative_to(w.out).as_posix()
    w.fingerprint(benchmark="landing", start=doc["start"], end=doc["end"], days=days)
    w.measurement_file(rel)
    holds = not doc["errors"]
    if holds:
        s = doc["summary"]
        detail = (f"every input read: {len(doc['rows'])} PRs landed by the loop from {doc['start']} to {doc['end']}, "
                  f"{s['plain']['n']} plain and {s['lean']['n']} Lean counted, {s['unmarked']} with no mark in force left out")
    else:
        detail = "not every input was read, so nothing is measured: " + "; ".join(doc["errors"])[:1500]
    w.validation("passed" if holds else "failed", "benchmarks/landing/landing.py", detail, [rel])
    if holds:
        for m in measurements(doc["summary"]):
            w.measurement(inputs=[f"{Q.CONTROL}:{Q.LIVE}/decisions.jsonl"], **m, mark="measured")
    return holds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landing.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="a checkout of the repository, fetched here: the heads' changes")
    ap.add_argument("--out", type=Path, required=True, help="where landing.json goes ($OUT under a research run)")
    ap.add_argument("--days", type=float, default=DAYS, help=f"the window's length in days (default {DAYS})")
    ap.add_argument("--until", metavar="YYYY-MM-DDTHH:MM:SSZ", help="the window's end (default now)")
    a = ap.parse_args(argv)
    try:
        end = Q.ts(a.until) if a.until else datetime.now(UTC).timestamp()
    except ValueError as e:
        print(f"landing: --until {a.until}: {e}; write a UTC time like 2026-01-31T08:00:00Z", file=sys.stderr)
        return 2
    if a.days <= 0:
        print(f"landing: --days {a.days} is not a length", file=sys.stderr)
        return 2
    start = end - timedelta(days=a.days).total_seconds()
    fetched = subprocess.run(["git", "-C", str(a.repo), "fetch", "-q", "origin", "main"], capture_output=True, text=True)
    if fetched.returncode:
        print(f"landing: git fetch origin main: {fetched.stderr.strip()[-300:]}; a head it lacks is unread", file=sys.stderr)
    doc = collect(a.repo, start, end)
    holds = write(kit.writer(a.out), doc, a.out, a.days)
    print(json.dumps({k: doc.get(k) for k in ("start", "end", "summary", "errors")}, indent=1))
    return 0 if holds else 1


if __name__ == "__main__":
    sys.exit(main())
