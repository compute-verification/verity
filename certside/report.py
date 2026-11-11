#!/usr/bin/env python3
"""The reading of a network certifier run's record (`warden_serve.py rehearse`, then `check`): counts and bucket
numbers, from three files of the run's `out/warden/` and nothing else.

    python3 certside/report.py OUT        # a run's out/ (or its out/warden); prints, and writes OUT/report.json

- `deployment.json`: the registered schedule (`WardenParams`), its digest, the slack and the run's windows;
- `warden.json`: the certifier's own log. It has one entry for each connection that got an ordinal, written when the
  connection ended or the run did: the bucket it arrived in (`arrived`: the first tick at or after the listener took
  it), its admission's (`admitted`: it got a lane), its request's END's (`ingress_end`: the release), its last
  cells' (`ended`), its egress cells, its failure's reason and egress cell (`failed`, `failed_at`), and `cut`, set
  when the run ended first. A refused connection has no ordinal and no entry: `refused` counts them and
  `refused_windows` names their windows;
- `check.json`: the offline check's verdict, its count of certified link-windows and the tick's lateness.

Standard library only, so it runs on a fetched record with any python3 from 3.7. It prints no token id, no byte of a request
or a response, no model's or user's name, and no address: every figure is a count, a bucket number or a time in ms.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

#: `warden.REASONS`, in its order; a reason the record names beyond these is counted under its own name
REASONS = ("late", "limit", "upstream", "ingress", "malformed", "format", "cap")
#: what a reader may want and the three files don't hold
NOT_IN_THE_RECORD = (
    "when the server's bytes for each egress cell reached the certifier: a connection's log has buckets, not arrival "
    "times, so the delay a connection needed (how far ahead of its deadline each cell's bytes were) can't be read "
    "here; `late` says only that one cell's bytes weren't there at its deadline",
    "a refused connection's arrival: it has no ordinal and no entry, only the count and the windows",
    "the wait for a lane to finer than a bucket: `arrived` is the first tick at or after the listener took the "
    "connection, so the wait is whole buckets, and -1 when the admitting tick itself ran past its boundary",
    "the advice bits: they are read from the server's own record, joined to this one by release order and bucket",
)


def distribution(values: list[int]) -> dict:
    """A list of bucket counts as its size, extremes, quantiles and histogram (value: connections)."""
    xs = sorted(values)
    pick = (lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]) if xs else (lambda q: None)
    return {"n": len(xs), "min": pick(0.0), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": pick(1.0),
            "histogram": {str(v): n for v, n in sorted(Counter(xs).items())}}


def kinds(problems: list[str]) -> dict[str, int]:
    """The check's problems by kind: each with its numbers replaced by N."""
    return dict(Counter(re.sub(r"\d+(\.\d+)?", "N", re.sub(r"\{.*\}", "{…}", p)) for p in problems).most_common())


def report(run: Path) -> dict:
    dep = json.loads((run / "deployment.json").read_text())
    log = json.loads((run / "warden.json").read_text())
    check = json.loads((run / "check.json").read_text())
    params, tau = dep["params"], dep["params"]["bucket_ms"]
    conns = log["connections"]
    admitted = [c for c in conns if c["admitted"] is not None]
    failed = [c for c in conns if c["failed"] is not None]
    reasons = Counter(c["failed"] for c in failed)
    late = [c for c in failed if c["failed"] == "late"]
    released = [c for c in admitted if c["ingress_end"] is not None]
    s = check["summary"]
    watermark = log.get("watermark")
    usage = log.get("usage") or {}
    return {
        "schedule": {**params, "delay_ms": params["delay"] * tau, "window_ms": params["window"] * tau},
        "digest": dep["digest"], "slack_ms": dep["slack_ms"], "windows": dep["windows"],
        "check": {"accepted": check["accepted"], "problems": len(check["problems"]),
                  "problems_listed_at_most": 50, "kinds": kinds(check["problems"])},
        "connections": {
            "arrived": log["ordinals"] + log["refused"], "with_an_ordinal": log["ordinals"], "logged": len(conns),
            "refused": log["refused"], "refused_windows": log["refused_windows"],
            "admitted": len(admitted),
            "released": len(released),
            "ended": sum(c["failed"] is None and c["ended"] is not None for c in conns),
            "failed": len(failed),
            "failed_by_reason": {r: reasons[r] for r in (*REASONS, *sorted(set(reasons) - set(REASONS)))},
            "cut_by_the_run_end": {"queued": sum(c["cut"] and c["admitted"] is None for c in conns),
                                   "served": sum(c["cut"] and c["admitted"] is not None for c in conns)},
        },
        "lane_wait_buckets": {**distribution([c["admitted"] - c["arrived"] for c in admitted]),
                              "waited": sum(c["admitted"] > c["arrived"] for c in admitted)},
        "ingress_buckets": distribution([c["ingress_end"] - c["admitted"] + 1 for c in released]),
        "late": {"connections": len(late), "failed_at_egress_cell": sorted(c["failed_at"] for c in late),
                 "histogram": {str(v): n for v, n in sorted(Counter(c["failed_at"] for c in late).items())}},
        "link_windows": {"certified": s["certified_windows"], "of": 3 * s["windows"],
                         "cell_rows_registered": s["cell_rows"]["registered"], "cell_rows_of": s["cell_rows"]["of"],
                         "withheld_windows": s["withheld_windows"], "signer_refused": len(s["signer_refused"])},
        "tick": {"lateness_ms": s["tick_lateness_ms"], "slack_ms": dep["slack_ms"],
                 "cells_in_their_bucket_at_the_client": s["cells_in_their_bucket"],
                 "ticker_stalls": s.get("ticker_stalls") and {k: s["ticker_stalls"].get(k) for k in ("episodes", "max_ms")},
                 "nice": usage.get("nice"), "cpus": len(usage.get("cpus") or [])},
        "watermark": watermark and {"last": watermark["last"], "lost": len(watermark["lost"]),
                                    "stopped": watermark["error"] and watermark["error"].split(":")[0]},
        "not_in_the_record": list(NOT_IN_THE_RECORD),
    }


def text(r: dict) -> str:
    p, c, t, w = r["schedule"], r["connections"], r["tick"], r["link_windows"]

    def dist(d: dict) -> str:
        return f"n {d['n']}, min {d['min']}, p50 {d['p50']}, p90 {d['p90']}, p99 {d['p99']}, max {d['max']}"

    lines = [
        "schedule [deployment.json]: " + ", ".join(f"{k} {v}" for k, v in p.items() if not k.endswith("_ms") or k == "bucket_ms"),
        f"  D = {p['delay']} buckets = {p['delay_ms'] / 1000:g} s; a window = {p['window_ms'] / 1000:g} s; {r['windows']} "
        f"windows; slack {r['slack_ms']:g} ms",
        f"  digest {r['digest']}",
        f"check [check.json]: {'ACCEPTED' if r['check']['accepted'] else 'REFUSED'}, {r['check']['problems']} problems"
        + (" (the check lists at most 50)" if r["check"]["problems"] >= 50 else ""),
        *(f"  {n} × {k}" for k, n in r["check"]["kinds"].items()),
        f"connections [warden.json]: arrived {c['arrived']} ({c['with_an_ordinal']} with an ordinal, {c['logged']} logged), "
        f"refused {c['refused']} (windows {c['refused_windows']}), admitted {c['admitted']}, released {c['released']}, "
        f"ended {c['ended']}, failed {c['failed']}, cut by the run's end {c['cut_by_the_run_end']['queued']} queued and "
        f"{c['cut_by_the_run_end']['served']} served",
        "  failed by reason: " + ", ".join(f"{k} {v}" for k, v in c["failed_by_reason"].items()),
        f"wait for a lane, buckets (admitted − arrived, the admitted): {dist(r['lane_wait_buckets'])}; "
        f"{r['lane_wait_buckets']['waited']} waited",
        f"ingress, buckets (ingress_end − admitted + 1, the released): {dist(r['ingress_buckets'])}",
        f"failed late: {r['late']['connections']}; the egress cell of each FAIL: {r['late']['failed_at_egress_cell']}",
        f"link-windows [check.json]: {w['certified']} of {w['of']} certified, cell rows {w['cell_rows_registered']} of "
        f"{w['cell_rows_of']} registered, withheld windows {w['withheld_windows']}, signer refused {w['signer_refused']}",
        f"tick [check.json]: lateness ms {t['lateness_ms']} against a slack of {t['slack_ms']:g} ms; cells in their bucket "
        f"at the client {t['cells_in_their_bucket_at_the_client']}; ticker stalls {t['ticker_stalls']}; the certifier ran "
        f"at nice {t['nice']} on {t['cpus']} cpus",
        f"watermark [warden.json]: {r['watermark']}",
        "not in the record:", *(f"  - {x}" for x in r["not_in_the_record"]),
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out = Path(argv[0])
    run = out / "warden" if (out / "warden" / "warden.json").exists() else out
    missing = [n for n in ("deployment.json", "warden.json", "check.json") if not (run / n).exists()]
    if missing:
        raise SystemExit(f"report: {run} has no {', '.join(missing)}: a run that never served has no record to read")
    r = report(run)
    (out / "report.json").write_text(json.dumps(r, indent=1) + "\n")
    print(text(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
