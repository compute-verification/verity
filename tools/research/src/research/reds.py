from __future__ import annotations

import re
from typing import Iterable

RATCHET_STEP = "ratchet-gate"
RATCHET_FAILED_RE = re.compile(r"^  (\S+) \(([0-9a-f]{12})\): failed: (.*)$", re.M)
RATCHET_PR_RE = re.compile(r"#(\d+)")
CIRCUIT_STEP = "circuit-check"
CIRCUIT_FAILED_RE = re.compile(r"^FAIL ([\w-]+/[\w-]+) (\S+?):?(?: (.*))?$", re.M)
CIRCUIT_KILLED = "circuit-check/killed"


def ratchet_failed(log: str) -> list[tuple[str, str, str]]:
    return RATCHET_FAILED_RE.findall(log)


def ratchet_prs(log: str) -> list[int]:
    return list(dict.fromkeys(int(m[1]) for name, _, _ in ratchet_failed(log) if (m := RATCHET_PR_RE.fullmatch(name))))


def circuit_failed(log: str) -> list[tuple[str, str, str]]:
    return CIRCUIT_FAILED_RE.findall(log)


def named(step: str, log: str) -> dict[str, str]:
    if step == RATCHET_STEP:
        return {name: f"{name} ({top}): {why}" for name, top, why in ratchet_failed(log)}
    if step == CIRCUIT_STEP:
        return {f"{code} {target}": f"{code} {target} {what}".rstrip() for code, target, what in circuit_failed(log)}
    return {}


def job_owner(jobs: Iterable[object], job: str) -> str | None:
    return next((t["owner"] for t in jobs if isinstance(t, dict) and t.get("name") == job and isinstance(t.get("owner"), str)),
                None)


def red_owners(lanes: Iterable[str | None], owners: Iterable[str | None], allowed_by: Iterable[str] = (),
               also: Iterable[str] = ()) -> tuple[str | None, ...]:
    return tuple(dict.fromkeys(h for h in (*lanes, *owners, *allowed_by, *also) if h)) or (None,)
