"""(c) GM-01's verdict reproduces: `global_match.json`'s verdict, its 8 checks (G1..G8 status + problem counts), X-09's verdict and
cross-request total, the alternate (sequence) criterion's verdict, the lag account and the concurrency facts.

`global_match.json` does NOT carry its inputs (the comparator reads `match/instances.jsonl` + the global Program, 0.6-5.8 GB; the file
pins them by sha256 under `impl.fold_load`), so the CPU-only reproduction is: the v2 candidate's `global_match.json` must state the
same verdict fields (candidate mode); in self-check mode the record is checked for internal consistency (verdict == PASS iff no failed
check; `match_summary.json.global_match.verdict` agrees; the fold-record shas are present).  STRONG on the PASS side (a check that
flips, a count that moves), WEAK as a negative oracle: no reference row fails a G-check.
"""
from __future__ import annotations

import json

from .. import resolver

NAME = "global_match_checks"
TIER = "T0"
STRENGTH = "weak"
INPUTS = ("global_match.json", "match_summary.json")

VERDICT_FIELDS = ("verdict", "failed_checks", "eos_truncated_requests", "rules_version", "within_step_criterion", "chronology_unit")


def applies(row: resolver.Row) -> str | None:
    return None if row.has("global_match.json") else "no global_match.json on this row (Match never reached GM-01)"


def fold_pins(gm: dict) -> dict | list | None:
    """The sha256 pins of the fold record GM-01 read.  `impl.fold_load` is keyed by the fold DIRECTORY, an absolute path on the machine
    that ran it ({"/workspace/.../match": {"source", "operand_refs_form", "program_sha256", "instances_sha256"}}), so the pins are one level
    down and the path does not enter: a candidate that read the same record elsewhere pins the same bytes.  One fold dir (what GM-01
    writes) -> its pins; several -> the sorted list of them.  The TOP level's keys are paths: no `*sha256` key is there."""
    fl = (gm.get("impl") or {}).get("fold_load") or {}
    pins = [p for p in ({k: v for k, v in n.items() if k.endswith("sha256")} for n in fl.values() if isinstance(n, dict)) if p]
    pins.sort(key=lambda p: json.dumps(p, sort_keys=True))
    return (pins[0] if len(pins) == 1 else pins) if pins else None


def lift(gm: dict) -> dict:
    out = {k: gm.get(k) for k in VERDICT_FIELDS}
    out["checks"] = {c["id"]: {"status": c.get("status"), "n_problems": c.get("n_problems", len(c.get("problems") or []))}
                     for c in (gm.get("checks") or []) if isinstance(c, dict) and c.get("id")}
    x = gm.get("x09") or {}
    out["x09"] = {"verdict": x.get("verdict"), "xreq_total": x.get("xreq_total"), "within_step_criterion": x.get("within_step_criterion"),
                  "unattributed": ((x.get("attribution") or {}).get("unattributed"))}
    alt = gm.get("alternate_verdict") or {}
    out["alternate_verdict"] = {"criterion": alt.get("criterion"), "verdict": alt.get("verdict"), "failed_checks": alt.get("failed_checks")}
    lag = gm.get("lag") or {}
    out["lag"] = {k: lag.get(k) for k in ("declared", "record_attests", "agree", "exercised", "variable_lag", "eos_stop_requests", "tokens_json_lag")}
    ac = gm.get("achieved_concurrency") or {}
    out["achieved_concurrency"] = {k: ac.get(k) for k in ("programmed_max", "observed_max", "declared_requests")}
    out["requests_n"] = len(gm.get("requests") or [])
    out["fold_record_pins"] = fold_pins(gm)
    return out


def expected(row: resolver.Row) -> dict:
    return lift(row.json("global_match.json"))


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("global_match.json")
    if cand is not None:
        return dict(lift(json.load(open(cand))), source="candidate")
    gm = row.json("global_match.json")
    out = dict(lift(gm), source="record")
    statuses = [v["status"] for v in out["checks"].values()]
    out["consistent"] = (out["verdict"] == "PASS") == (not out["failed_checks"] and all(s == "PASS" for s in statuses))
    if row.has("match_summary.json"):
        ms = row.json("match_summary.json")
        out["match_summary_agrees"] = ((ms.get("global_match") or {}).get("verdict") == out["verdict"]) and (ms.get("compare_verdict") in (None, out["verdict"]))
    return out
