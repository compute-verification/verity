"""Each `one_linear.sh` audit arm's outcome, judged from its status, its log and its ``summary.json``.

An arm's outcome is ``ACCEPT``, ``REFUSED`` (by the refusal's own name), ``REJECT`` (status 1 with no refusal the arm
names), ``ERROR`` (a crash, a panic, an undocumented status, no record) or ``FAIL`` (an accept the arm can't have). An arm passes only on what it expects: the
honest arm on ``ACCEPT``; a negative on ``REFUSED`` by the name it expects, or for ``forged-one`` on ``ACCEPT`` when its
forged unit's block is not drawn. A crash is never a refusal.

    one_linear_arms.py ARM STATUS LOG SUMMARY_JSON   # prints ``ARM <arm> <OUTCOME> <why>``; exits 0 when the arm passes
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: `records_audit.py audit`'s statuses: 0 passed, 1 not passed; it writes summary.json either way.
DOCUMENTED = (0, 1)
CRASH = re.compile(r"^Traceback \(most recent call last\)|panicked at|^PANIC|INTERNAL PANIC|Segmentation fault", re.M)
#: The prefix of `OneStageZk.SUBSET_ONLY`'s text, which names the sampling design's kind after it.
SUBSET_ONLY = "the sampling design is "
SUBSET_ONLY_TAIL = "which the one-statement audit's soundness statement doesn't cover"
#: `one_linear.sh`'s ``--forge 1``; a summary's ``draw`` lists the drawn units.
FORGED_UNIT = 1
EXPECTS = {"honest": "ACCEPT", "registered": "REFUSED", "forged-drawn": "REFUSED", "forged-one": "ACCEPT or REFUSED",
           "stratified": "REFUSED"}


#: `Record.checks`' refusal of evidence naming no session root, which an honest window whose ports are all held by
#: its members' registered reads meets: a completeness limit of the record, not a refusal of the window's values.
NO_ROOTS = ("REJECT at roots: the audit record refuses a window whose evidence names no session root "
            "(Record.checks' emptiness refusal); every port is held by the members check (headerHeldAt), so this "
            "is a completeness limit, not a wrong value")


def refusal(summary: dict) -> str | None:
    """Why ``summary`` is no accept, or None for an accept: the record's failed checks (`a5`'s ``checks_failed``)."""
    audit = summary.get("audit") or {}
    if summary.get("ok") is True and audit.get("accepted") is True:
        return None
    failed = audit.get("checks_failed") or {}
    if set(failed) == {"roots"} and not (failed["roots"] or {}).get("ports"):
        return NO_ROOTS
    if failed:
        return "REJECT at " + ", ".join(sorted(failed)) + ": " + json.dumps(failed, sort_keys=True, default=str)[:600]
    return "REJECT: " + json.dumps({"refused": summary.get("refused"), "why": audit.get("why")}, default=str)[:600]


def _window(log: str) -> dict:
    for line in log.splitlines():
        if line.startswith("STEP ") and '"step": "window"' in line:
            return json.loads(line[5:])
    return {}


def judge(arm: str, status: int, log: str, summary: dict | None) -> tuple[str, str, bool]:
    """The arm's outcome, why, and whether it is what the arm expects."""
    if arm not in EXPECTS:
        raise ValueError(f"no arm {arm!r}: one of {sorted(EXPECTS)}")
    if m := CRASH.search(log):
        return "ERROR", f"a crash: {log[m.start():].splitlines()[0][:300]}", False
    if status not in DOCUMENTED:
        return "ERROR", f"status {status}, which records_audit.py audit does not document", False
    if summary is None:
        return "ERROR", f"status {status} and no summary.json", False
    audit = summary.get("audit") or {}
    if status == 0 and summary.get("ok") is True:
        if arm == "forged-drawn":
            forged = summary.get("forged") or {}
            ok = forged.get("ok") is True
            return ("REFUSED", f"forged_outcome passes on {forged.get('passes_on')}", True) if ok else (
                "FAIL", "status 0 with no forged_outcome", False)
        if arm == "forged-one":
            w, draw = _window(log), summary.get("draw")
            if not w.get("units") or not w.get("blocks") or not isinstance(draw, list) \
                    or not all(isinstance(b, int) for b in draw):
                return "ERROR", "an accept whose window step or draw this judge can't read", False
            per = w["units"] // w["blocks"]
            block = FORGED_UNIT // per
            if block in {u // per for u in draw}:
                return "FAIL", f"accepted with the forged unit's block {block} drawn (units {draw})", False
            return "ACCEPT", f"the forged unit's block {block} is not drawn (units {draw})", True
        return "ACCEPT", "status 0, summary ok", arm == "honest"
    if status == 0:
        return "ERROR", "status 0 with a summary that is not ok", False
    text = json.dumps(summary, sort_keys=True, default=str)
    if arm == "stratified" and SUBSET_ONLY in text and SUBSET_ONLY_TAIL in text:
        return "REFUSED", f"SUBSET_ONLY: {text[text.index(SUBSET_ONLY):][:200]}", True
    members = audit.get("members_lean") or {}
    if arm == "registered" and members.get("ran") is True and members.get("ok") is False:
        return "REFUSED", f"the members check: {str(members.get('reason') or members.get('codes'))[:300]}", True
    if arm == "forged-one" and audit.get("accepted") is False:
        return "REFUSED", f"the audit record rejects: {json.dumps(audit.get('why'), default=str)[:300]}", True
    why = refusal(summary) or "REJECT"
    return "REJECT", f"status 1, not {EXPECTS[arm]} by its name: {why[len('REJECT'):].lstrip(': ')[:700]}", False


def main(argv: list[str]) -> int:
    arm, status, log, path = argv
    p = Path(path)
    summary = json.loads(p.read_text()) if p.is_file() else None
    outcome, why, passed = judge(arm, int(status), Path(log).read_text(errors="replace"), summary)
    print(f"ARM {arm} {outcome} {why}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
