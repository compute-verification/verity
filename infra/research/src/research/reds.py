"""What a failed step names, and whom its red rings, for the readers that share them: the verified queue's loop
(`research.vqueue_loop`, through `research.vqueue_slim`), drive, and the failed-checks monitor (`research.monitors`), which
imports this on the relay.  The ratchet gate's line for each merge it fails (`ratchet_failed`, `ratchet_prs`), circuit-check's
for each target (`circuit_failed`), both by step (`named`); the owner ci.toml gives a job (`job_owner`); and the order a red
rings its owners in (`red_owners`).  Stdlib only."""
from __future__ import annotations

import re
from typing import Iterable

#: the check step that runs the vLLM integration's ratchet gate on the merges above the check's base
RATCHET_STEP = "ratchet-gate"
#: the ratchet gate's line for a change it fails (`ratchet_gate.render`): its name (`#N`, or a range of commits), its top
#: commit's first 12 digits, and why
RATCHET_FAILED_RE = re.compile(r"^  (\S+) \(([0-9a-f]{12})\): failed: (.*)$", re.M)
#: a ratchet gate's name for one merge: the queue's `Queue: #N`
RATCHET_PR_RE = re.compile(r"#(\d+)")
#: the check step that runs `circuit-check --all`
CIRCUIT_STEP = "circuit-check"
#: circuit-check's line for a failure (`checks.Findings.fail`): `FAIL CHECK/CODE TARGET WHAT`, TARGET a Definition, template
#: or row with its statics, followed by a colon where circuit-check names its own failure (`circuit-check/killed`, `/error`)
CIRCUIT_FAILED_RE = re.compile(r"^FAIL ([\w-]+/[\w-]+) (\S+?):?(?: (.*))?$", re.M)
#: circuit-check's CHECK/CODE for a target whose process died, which is its machine's doing
CIRCUIT_KILLED = "circuit-check/killed"


def ratchet_failed(log: str) -> list[tuple[str, str, str]]:
    """(name, top commit's first 12 digits, why) of each change ratchet-gate's ``log`` says the gate failed."""
    return RATCHET_FAILED_RE.findall(log)


def ratchet_prs(log: str) -> list[int]:
    """The PR of each `#N` merge ratchet-gate's ``log`` says the gate failed, once each, in its order."""
    return list(dict.fromkeys(int(m[1]) for name, _, _ in ratchet_failed(log) if (m := RATCHET_PR_RE.fullmatch(name))))


def circuit_failed(log: str) -> list[tuple[str, str, str]]:
    """(CHECK/CODE, target, what) of each failure circuit-check's ``log`` names (`CIRCUIT_FAILED_RE`)."""
    return CIRCUIT_FAILED_RE.findall(log)


def named(step: str, log: str) -> dict[str, str]:
    """What failed step ``step``'s ``log`` names as failed, each by its key with its line: ratchet-gate's merges by name
    (`ratchet_failed`), circuit-check's failures by CHECK/CODE TARGET (`circuit_failed`); none of another step's."""
    if step == RATCHET_STEP:
        return {name: f"{name} ({top}): {why}" for name, top, why in ratchet_failed(log)}
    if step == CIRCUIT_STEP:
        return {f"{code} {target}": f"{code} {target} {what}".rstrip() for code, target, what in circuit_failed(log)}
    return {}


def job_owner(jobs: Iterable[object], job: str) -> str | None:
    """The owner ci.toml's job tables ``jobs`` give ``job``; None for one they don't declare (a preflight)."""
    return next((t["owner"] for t in jobs if isinstance(t, dict) and t.get("name") == job and isinstance(t.get("owner"), str)),
                None)


def red_owners(lanes: Iterable[str | None], owners: Iterable[str | None], allowed_by: Iterable[str] = (),
               also: Iterable[str] = ()) -> tuple[str | None, ...]:
    """Whom a red rings, first who acts: the lanes of the changes it names (``lanes``: its culprits', or its candidates'), then
    the owners of what failed (``owners``: a suite's or a target's, else its job's), then the owner of each allowance of it that
    expired (``allowed_by``), then ``also``; each once, and (None,) when none is known."""
    return tuple(dict.fromkeys(h for h in (*lanes, *owners, *allowed_by, *also) if h)) or (None,)
