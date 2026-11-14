"""The security review's serious findings still open, as a number: the findings of `findings.toml` beside this file that
neither a check this script runs nor a fresh reading closes, measured as `infra/security-findings-open`. Lower is better;
the goal is met at zero.

    research run --on auto --cwd source -- uv run python benchmarks/security_findings/open_findings.py

A finding with a `check` is decided by it (`CHECKS`), open or closed; one without is closed only by its `[finding.reading]`
while that names its command, reader, time and verdict and is less than `FRESH` old at the run's time; anything else is
open. The value's mark is `measured` only when every finding is decided by a check of the tree itself, and `partial`
while any rests on the tree's declared state of a machine, on a reading, or on nothing. It reads the checkout it runs in
and no machine, needs no secret, and writes under `--out` (`out/` outside a research run).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from research import addresses

from verity_bench import kit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIST = HERE / "findings.toml"
METRIC = "infra/security-findings-open"
#: how long a reading counts
FRESH = timedelta(days=7)
#: what a check reads: this repository's own files, or the tree's declared state of a machine
READS = ("tree", "declared")
READING = ("command", "reader", "at", "showed", "closed")
HOOKS = ".cursor/hooks.json"


@dataclass(frozen=True)
class Check:
    reads: str
    #: (closed, what it found in words) for the checkout at the root it is given
    run: Callable[[Path], tuple[bool, str]]


@dataclass(frozen=True)
class Status:
    id: str
    open: bool
    #: what decided it (`status`)
    basis: str
    why: str


def no_public_address(root: Path) -> tuple[bool, str]:
    found = addresses.scan(addresses.tracked(root), addresses.reader(root))
    holding = sum(any(kind == "ipv4" for kind, _ in hits) for hits in found.values())
    return holding == 0, f"{holding} tracked files hold a public IPv4 address"


def _starts_path(text: str, at: int) -> bool:
    """Whether a path relative to the checkout starts at `at` of a shell command: at the start of a word, after `./`,
    or right after a variable's or a substitution's `/`, which names no fixed place outside the checkout."""
    if at >= 2 and text[at - 2:at] == "./":
        at -= 2
    if at == 0 or text[at - 1] in " \t\n\"'=(;|&`":
        return True
    if text[at - 1] != "/":
        return False
    before = text[:at - 1]
    return before.endswith(("}", ")")) or bool(re.search(r"\$\w+$", before))


def runs_checkout_file(command: str, names: set[str]) -> bool:
    """Whether a hook's shell `command` names a file of the checkout: a path starting with one of the tree's top-level
    `names` where `_starts_path` says a relative path starts."""
    for name in names:
        for m in re.finditer(rf"{re.escape(name)}(?=[/\s\"';|&)]|$)", command):
            if _starts_path(command, m.start()):
                return True
    return False


def guard_outside_checkout(root: Path) -> tuple[bool, str]:
    hooks = json.loads((root / HOOKS).read_text(encoding="utf-8")).get("hooks", {})
    commands = [h["command"] for entries in hooks.values() for h in entries]
    names = {p.split("/", 1)[0] for p in addresses.tracked(root)}
    inside = sum(runs_checkout_file(c, names) for c in commands)
    return bool(commands) and inside == 0, f"{inside} of {len(commands)} hook commands of {HOOKS} run a file of the checkout"


CHECKS: Mapping[str, Check] = {
    "no-public-address": Check("tree", no_public_address),
    "guard-outside-checkout": Check("declared", guard_outside_checkout),
}


def _reading(r: object, now: datetime) -> tuple[str, bool, str]:
    """(basis, closed, why) of a finding's reading at the run's time `now`: taken (`reading`) only when it is whole and
    fresh, and closing the finding only when it says so."""
    if r is None:
        return "none", False, "no check and no reading"
    if not isinstance(r, dict):
        return "refused", False, "its reading is refused: it is a [finding.reading] table"
    if missing := [k for k in READING if k not in r or r[k] in ("", None)]:
        return "refused", False, f"its reading is refused: it lacks {', '.join(missing)}"
    if not all(isinstance(r[k], str) and r[k].strip() for k in ("command", "reader", "showed")):
        return "refused", False, "its reading is refused: command, reader and showed are words"
    if not isinstance(r["at"], datetime) or r["at"].tzinfo is None:
        return "refused", False, "its reading is refused: at is a UTC time such as 2026-01-01T00:00:00Z"
    if not isinstance(r["closed"], bool):
        return "refused", False, "its reading is refused: closed is true or false"
    age = now - r["at"]
    if age < timedelta(0):
        return "refused", False, "its reading is refused: it is later than the run"
    if age >= FRESH:
        return "none", False, f"its reading has expired: read {age.days} days before the run, and one counts under {FRESH.days}"
    return "reading", r["closed"], f"read by {r['reader']}: {r['showed']}"


def status(finding: Mapping, now: datetime, root: Path = ROOT, checks: Mapping[str, Check] = CHECKS) -> Status:
    """Whether `finding`, one [[finding]] of the list, is open at the run's time `now`, and what decided it: a check's
    `reads`, `reading`, `none` (nothing closes it) or `refused` (an entry the script can't take, which counts open)."""
    fid = str(finding.get("id", "?"))
    name = finding.get("check")
    if name is None:
        basis, closed, why = _reading(finding.get("reading"), now)
        return Status(fid, not closed, basis, why)
    if "reading" in finding:
        return Status(fid, True, "refused", "it has a check and a reading: a finding is decided one way")
    if name not in checks:
        return Status(fid, True, "refused", f"it names the check {name!r}, which open_findings.py doesn't run")
    check = checks[name]
    if finding.get("reads") != check.reads:
        return Status(fid, True, "refused", f"it says its check reads {finding.get('reads')!r}, and {name} reads {check.reads!r}")
    try:
        closed, why = check.run(root)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as e:
        return Status(fid, True, "refused", f"its check {name} couldn't run: {type(e).__name__}")
    return Status(fid, not closed, check.reads, why)


def mark(statuses: list[Status]) -> str:
    return "measured" if statuses and all(s.basis == "tree" for s in statuses) else "partial"


def load(path: Path = LIST) -> list[dict]:
    findings = tomllib.loads(path.read_text(encoding="utf-8")).get("finding", [])
    for f in findings:
        if not all(isinstance(f.get(k), str) and f[k].strip() for k in ("id", "statement", "how")):
            raise ValueError(f"{path}: each [[finding]] has an id, a statement and how it is checked, in words: {f.get('id')!r}")
    if len({f["id"] for f in findings}) != len(findings):
        raise ValueError(f"{path}: two [[finding]] entries share an id")
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    statuses = [status(f, datetime.now(UTC)) for f in load()]
    n, m = sum(s.open for s in statuses), mark(statuses)
    print(f"security findings open: {n} of {len(statuses)} (mark {m})")
    for s in statuses:
        print(f"  {s.id}: {'open' if s.open else 'closed'} ({s.basis}): {s.why}")
    w = kit.writer(out)
    w.measurement(METRIC, n, "findings", target=0, mark=m, inputs=[str(LIST.relative_to(ROOT))])
    refused = [s.id for s in statuses if s.basis == "refused"]
    w.validation("failed" if refused else "passed", "benchmarks/security_findings/open_findings.py",
                 f"entries refused: {', '.join(refused)}" if refused else f"read {len(statuses)} findings")


if __name__ == "__main__":
    main()
