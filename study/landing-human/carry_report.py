"""What `carry_study.py`'s verdicts mean for landing: how many re-verdicts carry, the mark-to-verdict hours they'd have saved,
the blocking heads they'd have dissolved, and the cases that test a false positive (a hold at the carried head, a revert of
the PR on `checked`).

    python3 study/landing-human/carry_report.py RESULTS.json LABELS.json BLOCKERS.json [MARKS_RESULTS.json]"""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def T(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours(a: str, b: str) -> float:
    return (T(b) - T(a)).total_seconds() / 3600


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else 0.0


def main() -> None:
    res = json.load(open(sys.argv[1]))
    labels = json.load(open(sys.argv[2]))
    blockers = json.load(open(sys.argv[3]))
    holds = {(r["pr"], r["head"]) for r in labels if r["key"] == "reviewed" and r["value"] == "hold"}
    reverts = collections.Counter(int(n) for s in subprocess.run(
        ["git", "log", "origin/checked", "--since=2026-10-22", "--format=%s"], cwd=REPO, capture_output=True, text=True).stdout.splitlines()
        if s.startswith(("Revert", "revert")) for n in re.findall(r"#(\d+)", s))
    out: dict = {"n": len(res)}
    for mode, get in (("strict", lambda r: r), ("wide", lambda r: r.get("wide") or r)):
        car = [r for r in res if get(r).get("carries")]
        saved = [max(0.0, hours(r["mark"], r["ts"])) for r in car if r.get("mark")]
        out[mode] = {
            "carry": len(car), "kinds": dict(collections.Counter(get(r)["kind"] for r in res).most_common()),
            "by_reviewer": dict(collections.Counter(r["by"] for r in car).most_common()),
            "mark_to_verdict_h": {"sum": round(sum(saved)), "p50": round(q(saved, .5), 2), "p90": round(q(saved, .9), 2)},
            "held_at_carried_head": sorted({r["pr"] for r in car if (r["pr"], r["new"]) in holds}),
            "reverted_on_checked": sorted({r["pr"] for r in car if reverts[r["pr"]]}),
        }
        idx = {(r["pr"], r["new"][:8]): get(r) for r in res}
        bl = [b for b in blockers if idx.get((b["pr"], b["sha"][:8]), {}).get("carries")]
        out[mode]["blockers"] = {
            "carried": [f"#{b['pr']}@{b['sha'][:8]} ({b['max_waiting']} behind, {b['mark_to_end_h']} h)" for b in bl],
            "of": len(blockers), "re_verdicts": sum(1 for b in blockers if (b["pr"], b["sha"][:8]) in idx),
            "pr_hours_mark_to_end": round(sum(b["mark_to_end_h"] * b["max_waiting"] for b in bl)),
            "pr_hours_seen_to_end": round(sum(b["lo_h"] * b["max_waiting"] for b in bl)),
        }
    if "later" in json.dumps(res[:50]) or any("later" in r for r in res):
        out["untrusted_parent_carry_once_reviewed"] = sum(1 for r in res if (r.get("later") or {}).get("carries"))
    if len(sys.argv) > 4:
        marks = json.load(open(sys.argv[4]))
        out["re_heads"] = {"n": len(marks), "strict": dict(collections.Counter(r["kind"] for r in marks).most_common()),
                           "wide": dict(collections.Counter((r.get("wide") or r)["kind"] for r in marks).most_common())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
