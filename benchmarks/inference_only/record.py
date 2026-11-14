"""A served run's record, read for `report.py --served`: the inference-only check on the run's files, what it costs in
the clear, and the share of a block's work that no PoUW check credits.

* **The files.** The run's `out/` (`OUT_FILES`): its compose's verdicts, its policy, the rate rule's limits and the
  count's. The Program, the circuit record and the registration log the run registered are named apart, since a run's
  record keeps them outside `out/`.
* **The check** is the policy's record check, read through its command lines: its script writes the case
  (`served_record_case.py --case-only`) and its compiled command judges it (`inference-only-record-check`, built with
  `lake`), each timed in the clear (`built.timed`). The command prints each verdict's whole detail, the private
  record's. Kept here is the public form alone (`public`), as `PUBLIC` names the Lean function giving it beside each
  line's `checkDetail`: `{complies}`, or the refusal's `error` and `message`. The case and the detail are written to a
  scratch directory outside the run's and deleted with it.
* **The share** is PoUW's alone (`share`): s = 1 − C / W, C the MACs PoUW's verdict of record credits the run's
  blocks (`quantity.work.credited`) and W their work in the policy's unit (`class.slots` × `n_inputs.work.theta_macs`).
  C is read only where `VERDICT` is `accept`. A development accept (`missing`, with its `development` piece) is no
  accept, and with none s = 1. A verdict whose MACs are not `class.slots` × the policy's linears, or whose C is below 0
  or above W or its own MACs, is refused. What ties the verdict to the record the check read: the window record the
  verdict holds its passes to (`quantity.tie.record`) is the `previous` of the run's registration, the log's last
  record, whose chain the check holds; another is refused.
* **The class under the tie** (`tied_of`): the record's policy with the linears' outputs taken out of its k and
  `tied.toml`'s `[fresh]` put in, by the policy's own k by kind (`k_by_kind`, its keys `KINDS`), whose kinds must be
  the linears' outputs (`OUTPUTS`) and the others `tied.toml`'s source names, exactly.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import built as B
import policy as P


def _declared() -> str:
    """The record check as its package declares it (`lakefile.toml`): the executable whose root is `RecordCheckMain`."""
    lake = tomllib.loads((B.PACKAGE / "lakefile.toml").read_text())
    (lib,) = (x["name"] for x in lake["lean_lib"])
    (exe,) = (x["name"] for x in lake["lean_exe"] if x["root"] == f"{lib}.RecordCheckMain")
    return exe


CHECK = _declared()
#: the check's script that writes a case from a run's files
SCRIPT = B.PACKAGE / "scripts" / "served_record_case.py"
#: each line the check prints, its rule, and the Lean function that gives its public form beside its `checkDetail`
PUBLIC = {"check": ("the rate rule", "NciCheck.Record.check"), "count": ("the count", "NciCheck.RecordCount.check")}
LINE = re.compile(r"\[(\d+) s\] (read the case|check: |count: )(.*)")
#: the files of the run's `out/` read, as each is named there
OUT_FILES = {"verdicts": "record/verdicts.json", "policy": "inference-only-policy.json",
             "limits": "inference-only-limits.json", "count": "block-counts-*.json"}
POLICY_FORMAT = "verity-bench/inference-only-policy/v1"
#: the field that says whether PoUW's verdict is of record
VERDICT = "protocols.work.verdict"
#: the kind of a record policy's k that the tie takes out
OUTPUTS = "the outputs of the linears"
#: a record policy's kinds of k by its writer's keys (`inference_only.KINDS` in benchmarks/one_stage), each by the name
#: `policy.stated` gives that part
KINDS = {"linears": OUTPUTS, "scan_state": "the scan state row"}
NO_TIE = "nothing ties the PoUW verdict to the record the check read"
NO_KINDS = ("the record's policy states k as one number, with no k by kind (k_by_kind), so the linears' outputs the tie "
            "takes out of k can't be read from it: it was written before its writer (benchmarks/one_stage/"
            "inference_only.py) stated one, and there is no figure for the tied class")


class Refused(Exception):
    """The record is refused; the message says why."""


@dataclass(frozen=True)
class Files:
    """A run's `out/`, and the Program, circuit record and registration log it registered."""

    out: Path
    program: Path
    circuit: Path
    logs: tuple[Path, ...]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: can't be read as JSON ({type(e).__name__}: {e})") from None


def read(files: Files) -> dict[str, Any]:
    """The record's files (`OUT_FILES`): its verdicts, its policy, and the limits each rule is judged against, the count's
    `(c, k, n)` the policy's own. A file that isn't there is named as one the run did not write."""
    counts = sorted(files.out.glob(OUT_FILES["count"]))
    if len(counts) > 1:
        raise Refused(f"{files.out}: a record holds one {OUT_FILES['count']}, the count's limits; it holds {len(counts)}")
    named = {"Program": files.program, "circuit record": files.circuit,
             **{f"record {k} of a registration log": p for k, p in enumerate(files.logs)},
             "compose verdicts": files.out / OUT_FILES["verdicts"], "policy": files.out / OUT_FILES["policy"],
             "rate rule's limits": files.out / OUT_FILES["limits"],
             "count's limits": counts[0] if counts else files.out / OUT_FILES["count"]}
    for what, path in named.items():
        if not path.is_file():
            raise Refused(f"the run wrote no {what} ({path})")
    verdicts, policy = (_json(files.out / OUT_FILES[x]) for x in ("verdicts", "policy"))
    count, limits = _json(counts[0]), files.out / OUT_FILES["limits"]
    if policy.get("format") != POLICY_FORMAT:
        raise Refused(f"{OUT_FILES['policy']}: format {policy.get('format')!r}, not {POLICY_FORMAT}")
    try:
        cls, work = policy["class"], policy["n_inputs"]["work"]
        whole = [policy[x] for x in "ckn"] + [cls[x] for x in ("slots", "prompt", "outputs")] + [
            work[x] for x in (*P.WORK, "theta_macs")]
    except (KeyError, TypeError) as e:
        raise Refused(f"{OUT_FILES['policy']}: no {e}") from None
    if not all(type(x) is int and x > 0 for x in whole):
        raise Refused(f"{OUT_FILES['policy']}: c, k, n, the class and the work's MACs are positive integers")
    if not isinstance((policy.get("serve") or {}).get("model"), str):
        raise Refused(f"{OUT_FILES['policy']}: no serve.model")
    if work["theta_macs"] != work["linears"] + 2 * (work["attention"] + work["lm_head"]):
        raise Refused(f"{OUT_FILES['policy']}: theta_macs is not linears + 2 × (attention + lm_head)")
    if [count.get(x) for x in "ckn"] != [policy[x] for x in "ckn"]:
        raise Refused(f"{counts[0].name}: the count's limits {[count.get(x) for x in 'ckn']} are not the policy's c, k, n")
    return {"run": verdicts.get("run"), "verdicts": verdicts, "policy": policy, "limits": limits, "count": counts[0]}


def previous(files: Files) -> Any:
    """The `previous` of the run's registration, the log's last record: the record the run's registration follows."""
    return _json(files.logs[-1]).get("previous")


def share(verdicts: Mapping[str, Any], policy: Mapping[str, Any], before: Any) -> dict[str, Any]:
    """The share of the run's blocks' work that no PoUW check credits, from its verdict of record (module docstring),
    ``before`` the run's registration's `previous`."""
    row = (verdicts.get("protocols") or {}).get("work")
    if not isinstance(row, dict):
        raise Refused(f"the record's verdicts hold no PoUW row ({VERDICT.rsplit('.', 1)[0]})")
    slots, work = policy["class"]["slots"], policy["n_inputs"]["work"]
    W, verdict = slots * work["theta_macs"], row.get("verdict")
    said = {"field": VERDICT, "verdict": verdict, "answer_from": row.get("answer_from"), "W": W, "slots": slots,
            "theta_macs": work["theta_macs"]}
    if verdict != "accept":
        return {**said, "share": 1.0, "C": None, "development": "development" in row, "tie": None,
                "why": str(row.get("why") or row.get("reason") or "it gives no reason")}
    q = row.get("quantity") or {}
    C, macs = (q.get("work") or {}).get("credited"), (q.get("work") or {}).get("macs")
    if type(C) is not int or type(macs) is not int:
        raise Refused(f"PoUW's accept of record states no whole credited MACs and MACs (quantity.work): {C!r}, {macs!r}")
    if macs != slots * work["linears"]:
        raise Refused(f"PoUW's accept counts {macs:,} MACs of linears, not the record's class.slots × "
                      f"n_inputs.work.linears = {slots} × {work['linears']:,} = {slots * work['linears']:,}")
    if C < 0 or C > W:
        raise Refused(f"PoUW's accept credits C = {C:,} MACs, {'below 0' if C < 0 else f'above W = {W:,}'}")
    if C > macs:
        raise Refused(f"PoUW's accept credits C = {C:,} MACs, above the {macs:,} MACs of linears it counts")
    window = (q.get("tie") or {}).get("record")
    if window is None:
        tie = {"window": None, "previous": before, "why": NO_TIE}
    elif window != before:
        raise Refused(f"PoUW's verdict holds its passes to window record {str(window)[:16]}..., but the run's "
                      f"registration follows {str(before)[:16]}...: the verdict is about another record than the check's")
    else:
        tie = {"window": window, "previous": before, "why": None}
    return {**said, "share": 1 - C / W, "C": C, "macs": macs, "development": False, "tie": tie, "why": None}


def public(detail: Any) -> dict[str, Any]:
    """A line's public form, as its Lean function (`PUBLIC`) gives it from the same verdict: `{complies}`, or the
    refusal's `error` and `message`."""
    if isinstance(detail, dict) and isinstance(detail.get("error"), str):
        return {x: detail[x] for x in ("error", "message") if x in detail}
    if isinstance(detail, dict) and isinstance(detail.get("complies"), bool):
        return {"complies": detail["complies"]}
    raise RuntimeError(f"{CHECK} printed a verdict that is neither a judgment nor a refusal")


def checked(files: Files, got: Mapping[str, Any], scratch: Path, clock: Callable[[], float],
            exe: Path | None = None) -> dict[str, Any]:
    """The case of the run's files, written by `SCRIPT`, and `CHECK`'s two verdicts on it (``exe``, else built here)
    in their public form, with what each step cost; everything written under ``scratch``."""
    argv = [sys.executable, str(SCRIPT), "--program", str(files.program), "--record", str(files.circuit),
            "--log", *map(str, files.logs), "--limits", str(got["limits"]), "--count", str(got["count"]),
            "--out", str(scratch), "--name", "served", "--case-only"]
    written = B.timed(argv, scratch / "case.out", clock, private=True)
    case = scratch / "served.case.json"
    cost = B.timed([str(exe or B.lake_built(CHECK)), str(case)], scratch / "check.out", clock, private=True)
    said, at = {}, {}
    for line in (scratch / "check.out").read_text().splitlines():
        if m := LINE.fullmatch(line):
            what = m[2].removesuffix(": ") if m[2] != "read the case" else "read"
            at[what] = int(m[1])
            if what in PUBLIC:
                said[what] = public(json.loads(m[3]))
    if set(said) != set(PUBLIC) or "read" not in at:
        raise RuntimeError(f"{CHECK} printed no line for {', '.join(sorted({*PUBLIC, 'read'} - {*said, *at}))}")
    return {**said, "seconds": at, "cost": {"case": {**written, "bytes": case.stat().st_size}, "check": cost}}


def tied_of(policy: Mapping[str, Any], run: str | None, path: Path = P.TIED) -> P.Tied:
    """The record's class under the tie: ``policy``'s c, n and work, and its k by kind less `OUTPUTS` plus ``path``'s
    `[fresh]`; ``path``'s `[untied]` bound. Its kinds must be `OUTPUTS` and ``path``'s others, exactly."""
    t = P.tied(path)
    fresh = tomllib.loads(Path(path).read_text()).get("fresh") or {}
    want = {kind for kind, _, _ in t.policy.parts} - set(fresh)
    keyed = policy.get("k_by_kind")
    if keyed is None:
        raise Refused(NO_KINDS)
    if not (isinstance(keyed, dict) and all(type(b) is int and b >= 0 for b in keyed.values())):
        raise Refused("the record's policy's k_by_kind is an object of kinds, each its bits")
    if {KINDS.get(key) for key in keyed} != want | {OUTPUTS}:
        raise Refused(f"the record's policy's k is made of {sorted(keyed)}; under the tie it is the kinds "
                      f"{sorted(k for k, name in KINDS.items() if name in want | {OUTPUTS})}: {OUTPUTS!r} and the kinds "
                      f"{path.name}'s source names but its [fresh], {sorted(want)}, exactly")
    kinds = {KINDS[key]: bits for key, bits in keyed.items()}
    if sum(kinds.values()) != policy["k"]:
        raise Refused(f"the record's policy's kinds sum to {sum(kinds.values()):,} bits, not its k = {policy['k']:,}")
    cls = tuple(policy["class"][x] for x in ("slots", "prompt", "outputs"))
    if (positions := policy["class"].get("positions")) != P.class_of(cls).positions:
        raise Refused(f"the record's class {','.join(map(str, cls))} has {positions} positions, not the integration's "
                      f"{P.class_of(cls).positions}")
    work = {**{x: policy["n_inputs"]["work"][x] for x in P.WORK}, P.TOTAL: policy["n_inputs"]["work"]["theta_macs"]}
    parts = (*((kind, bits, "the record's policy's") for kind, bits in kinds.items() if kind != OUTPUTS),
             *((kind, bits, f"{Path(path).name}'s") for kind, bits in fresh.items()))
    k = policy["k"] - kinds[OUTPUTS] + sum(fresh.values())
    source = (f"run {run}'s policy with every tied value right: its c, n and work, and k its {', '.join(sorted(want))} "
              f"and {Path(path).name}'s {', '.join(fresh)}")
    p = P.Policy("record", policy["serve"]["model"], cls, policy["c"], k, policy["n"], work, parts, 0, source,
                 {"verdict": "accepted"})
    return dataclasses.replace(t, policy=p)
