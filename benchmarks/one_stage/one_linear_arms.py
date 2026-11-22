"""`one_linear.sh`'s honest audit's outcome, judged from its status, its log and its ``summary.json``.

The outcome is ``ACCEPT`` (status 0 and a summary that is ok), ``REJECT`` (status 1, by the refusal the summary or the
audit's log names) or ``ERROR`` (a crash, a panic, an undocumented status, no record, or status 0 with a summary that
is not ok). It passes only on ``ACCEPT``. A crash is never an accept, nor a refusal.

    one_linear_arms.py ARM STATUS LOG SUMMARY_JSON [LOG...]   # prints ``ARM <arm> <OUTCOME> <why>``; exits 0 on ACCEPT

Every LOG is read whole (the arm's own log holds only the tail of its audit's stderr), and a missing one is skipped.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: `records_audit.py audit`'s statuses: 0 passed, 1 not passed; it writes summary.json either way.
DOCUMENTED = (0, 1)
CRASH = re.compile(r"^Traceback \(most recent call last\)|panicked at|^PANIC|INTERNAL PANIC|Segmentation fault", re.M)
ARMS = ("honest",)


def refusal(summary: dict) -> str | None:
    """Why ``summary`` is no accept, or None for an accept: the record's failed checks (`a5`'s ``checks_failed``)."""
    audit = summary.get("audit") or {}
    if summary.get("ok") is True and audit.get("accepted") is True:
        return None
    failed = audit.get("checks_failed") or {}
    if failed:
        return "REJECT at " + ", ".join(sorted(failed)) + ": " + json.dumps(failed, sort_keys=True, default=str)[:600]
    return "REJECT: " + json.dumps({"refused": summary.get("refused"), "why": audit.get("why")}, default=str)[:600]


def _checks_before_draw(log: str) -> dict:
    """The checks a5 printed when it refused before the draw (its ``A5 {"refused_before_draw": ...}`` line in the audit's
    log), which records_audit's summary keeps only as booleans. The arm's own log holds a prefix of the line, cut
    short, so the longest copy is the whole one."""
    lines = [line for line in log.splitlines() if line.startswith('A5 {"refused_before_draw"')]
    return (json.loads(max(lines, key=len)[3:]).get("checks") or {}) if lines else {}


def judge(arm: str, status: int, log: str, summary: dict | None) -> tuple[str, str, bool]:
    """The audit's outcome, why, and whether it passes (an accept)."""
    if arm not in ARMS:
        raise ValueError(f"no arm {arm!r}: one_linear.sh runs {list(ARMS)}")
    if m := CRASH.search(log):
        return "ERROR", f"a crash: {log[m.start():].splitlines()[0][:300]}", False
    if status not in DOCUMENTED:
        return "ERROR", f"status {status}, which records_audit.py audit does not document", False
    if summary is None:
        return "ERROR", f"status {status} and no summary.json", False
    if status == 0:
        return ("ACCEPT", "status 0, summary ok", True) if summary.get("ok") is True else (
            "ERROR", "status 0 with a summary that is not ok", False)
    failed = {k: v.get("codes") for k, v in (summary.get("checks") or _checks_before_draw(log)).items()
              if isinstance(v, dict) and v.get("ok") is False}
    if failed and not (summary.get("audit") or {}).get("checks_failed"):
        return "REJECT", f"before the draw at {json.dumps(failed, sort_keys=True, default=str)[:700]}", False
    why = refusal(summary) or "REJECT"
    return "REJECT", f"status 1: {why[len('REJECT'):].lstrip(': ')[:700]}", False


def main(argv: list[str]) -> int:
    arm, status, log, path, *more = argv
    p = Path(path)
    summary = json.loads(p.read_text()) if p.is_file() else None
    text = "\n".join(Path(f).read_text(errors="replace") for f in (log, *more) if Path(f).is_file())
    outcome, why, passed = judge(arm, int(status), text, summary)
    print(f"ARM {arm} {outcome} {why}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
