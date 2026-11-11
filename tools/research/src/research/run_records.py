"""A run's own records, cut down to what its three times need: `research run-times` reads a runs directory with this file, on
the machine that holds it (sent as the stdin of `python3 -`) or here.

    python3 run_records.py RUNS_DIR FIRST LAST

prints one JSON line (`record`) for each run directory of RUNS_DIR whose id sorts in [FIRST, LAST] (an id, `rYYYYMMDD-HHMMSS-xxxx`,
starts with its launch time), then the line `{"read": N}`, so an answer cut short is seen as cut short.

No command line leaves the machine: a record says only what its command is (`kind`), which check slots it asks for (`pool`) and
for whom (`by`), read from `job.json`. From `stderr.log` it takes only slot.py's own headline (`slot: WHY; retrying every N s`),
never the lines under it, which quote other runs' commands. A record file that is there and can't be read, or isn't the JSON it
should be, is named in the run's record (`unread`), never read as absent. Stdlib only and no import from `research`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: a run directory's name
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}$")
#: slot.py's line when what it waits behind changes (tools/verity/check/slot.py)
WAITS = re.compile(r"^slot: (.+); retrying every \d+ s$", re.M)
#: how much of stderr.log is read for it (slot.py writes there before the job does)
STDERR_HEAD = 1 << 18
KINDS = ("quick suites", "Lean records", "check shards", "GPU jobs", "other")
#: a pool's slots file in a command (research.pods.nebius.slot_only.POOL, with the pool's name captured)
POOL = re.compile(r"CHECK_SLOTS=/workspace/research/locks/([A-Za-z0-9_-]+)/slots")
SLOT_PY, OWN_POOL, NO_POOL = "tools/verity/check/slot.py", "check", "-"
STATUS = ("state", "started_utc", "ended_utc", "rc", "timed_out")
SLOT = ("state", "since", "took", "left", "slot", "why", "priority", "twin")
REFUSED = ("at_utc", "fault", "reason")
CANCEL = ("t_utc", "actor", "reason")


def kind(command: list[str]) -> str:
    """The kind of task ``command`` runs, one of `KINDS`. A job that is a script of its own counts as other, whatever the
    script runs."""
    text = " ".join(command)
    if "gpu-lease" in text:
        return "GPU jobs"
    if "tools/verity/check/check.py" in text:
        return "check shards"
    if "lean_changed.py" in text and "--records" in command:
        return "Lean records"
    if "quick_tier.py" in text or ("tools/verity/check/suites.py" in text and "--quick" in text):
        return "quick suites"
    return "other"


def pool(command: list[str]) -> str:
    """The check slots ``command`` asks for: a pool's by its name, the machine's own as `OWN_POOL`, none as `NO_POOL`."""
    text = " ".join(command)
    if m := POOL.search(text):
        return m[1]
    return OWN_POOL if SLOT_PY in text else NO_POOL


def asked_by(command: list[str]) -> str | None:
    """Whom slot.py was told the run is for (its `--by`)."""
    at = command.index("--by") if "--by" in command else -1
    return command[at + 1] if 0 <= at < len(command) - 1 else None


def _text(p: Path, unread: dict[str, str], head: int = -1) -> str | None:
    """File ``p``'s text (its first ``head`` bytes), None where there is no such file. One that is there and can't be read is
    named in ``unread`` with why, and reads as None too."""
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            return f.read(head).decode(errors="replace")
    except OSError as e:
        unread[p.name] = f"{type(e).__name__}: {e.strerror or e}"
        return None


def _doc(p: Path, unread: dict[str, str]) -> dict:
    """``p``'s JSON object, {} where there is no such file. One that is no JSON object (a record cut short) is named in
    ``unread`` with why, and reads as {} too."""
    text = _text(p, unread)
    if text is None:
        return {}
    try:
        d = json.loads(text)
    except ValueError as e:
        unread[p.name] = f"not JSON: {e}"
        return {}
    if not isinstance(d, dict):
        unread[p.name] = "not a JSON object"
        return {}
    return d


def _cancel(d: Path, unread: dict[str, str]) -> dict:
    """The run's first cancel as `events.jsonl` has it: when, by what (`manual`, `timeout`) and the reason given."""
    for line in (_text(d / "events.jsonl", unread) or "").splitlines():
        if '"cancel_intent"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError as err:
            unread["events.jsonl"] = f"a cancel_intent line is not JSON: {err}"
            continue
        if isinstance(e, dict) and e.get("kind") == "cancel_intent":
            return {k: e[k] for k in CANCEL if e.get(k) is not None}
    return {}


def _behind(d: Path, unread: dict[str, str]) -> str | None:
    """What slot.py last said the run waits behind, from the head of its stderr.log."""
    said = WAITS.findall(_text(d / "stderr.log", unread, STDERR_HEAD) or "")
    return said[-1] if said else None


def record(d: Path) -> dict:
    """Run directory ``d`` as {run, machine, kind, pool, by, claimed, status, slot, refused, cancel, job_started, unread}, each
    key only where its record has it: a run with no `job.json` (a refused launch) has no kind, pool or machine. `slot.why` is
    slot.json's, else (a record written before slot.py kept it once the slot is taken) what stderr.log says. `unread` names
    each record file that is there and couldn't be read, with why: the rest of the record is from the files that could."""
    unread: dict[str, str] = {}
    job, status, slot, launch, refused, live = (_doc(d / n, unread) for n in (
        "job.json", "status.json", "slot.json", "launch.json", "refused.json", "check-live.json"))
    command = [str(a) for a in (job.get("execution") or {}).get("command") or []]
    slot = {k: slot[k] for k in SLOT if slot.get(k) is not None}
    if slot and not slot.get("why") and (said := _behind(d, unread)):
        slot["why"] = said
    out = {"run": d.name, "machine": ((job.get("run") or {}).get("remote") or {}).get("machine"),
           **({"kind": kind(command), "pool": pool(command), "by": asked_by(command)} if command else {}),
           "claimed": launch.get("claimed_utc"), "status": {k: status[k] for k in STATUS if status.get(k) is not None},
           "slot": slot, "refused": {k: refused[k] for k in REFUSED if refused.get(k) is not None},
           "cancel": _cancel(d, unread) if status else {}, "job_started": live.get("started_utc"), "unread": unread}
    return {k: v for k, v in out.items() if v not in (None, {}, [])}


def window(root: Path, first: str, last: str) -> list[Path]:
    """The run directories of ``root`` whose ids sort in [``first``, ``last``], oldest first."""
    try:
        names = sorted(p.name for p in root.iterdir() if RUN_ID.match(p.name) and first <= p.name <= last)
    except OSError as e:
        raise SystemExit(f"run_records: {root}: {e}") from None
    return [root / n for n in names]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    dirs = window(Path(argv[0]), argv[1], argv[2])
    for d in dirs:
        print(json.dumps(record(d)), flush=False)
    print(json.dumps({"read": len(dirs)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
