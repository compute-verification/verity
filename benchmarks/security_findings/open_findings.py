"""The security review's serious findings still open, as a number: the findings of `findings.toml` beside this file that no
check this script runs closes, measured as `infra/security-findings-open`. Lower is better; the goal is met at zero.

    research run --on auto --cwd source -- uv run python benchmarks/security_findings/open_findings.py

Only a check closes a finding: the one `CHECKS` holds for its id, which the list must name, run on the checkout at the commit
it is at. A finding with no check is open whatever the list says, and the list says in one line what check would close it.
The value's mark is `measured` only when every finding is decided by a check of the thing itself (`tree`), and `partial`
while any has no check or is checked on the tree's declared state (`declared`). It reads no machine, needs no secret, and
writes under `--out` (`out/` outside a research run).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from research import addresses

from verity_bench import kit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIST = HERE / "findings.toml"
METRIC = "infra/security-findings-open"
#: what a check reads: this repository's own files, the thing the finding is about, or the tree's declared state of a VM
READS = ("tree", "declared")
#: why an address the rule counts is excepted from finding 6: nobody's machine of ours
EXCEPT_KINDS = ("version", "example", "third party")
#: what a [[finding]] may hold, and nothing else
KEYS = {"id", "statement", "check", "reads", "how", "would_close", "except"}
HOOKS = ".cursor/hooks.json"
#: the one place the guard's cure runs it from, outside every checkout
GUARD_ROOT = "/opt/agent-guard/"
_GUARD_FILE = re.compile(re.escape(GUARD_ROOT) + r"[A-Za-z0-9_-][A-Za-z0-9_.-]*")
#: a word with nothing the shell expands, quotes, joins or redirects, and no path
_PLAIN = re.compile(r"[A-Za-z0-9_.,:=+%@-]+")


@dataclass(frozen=True)
class Check:
    name: str
    reads: str
    #: (closed, what it found in words) for the checkout at the root it is given and the finding's entry in the list
    run: Callable[[Path, Mapping], tuple[bool, str]]


@dataclass(frozen=True)
class Status:
    id: str
    open: bool
    #: what decided it: a check's `reads`, `none` (no check) or `refused` (an entry the script can't take, which counts open)
    basis: str
    why: str


def excepted(finding: Mapping) -> dict[str, int]:
    """{path: how many of its public addresses are excepted} from the finding's [[finding.except]] entries, each a path, a
    count, a kind of `EXCEPT_KINDS` and a reason."""
    out: dict[str, int] = {}
    for e in finding.get("except", []):
        if not (isinstance(e.get("path"), str) and isinstance(e.get("count"), int) and e["count"] > 0
                and e.get("kind") in EXCEPT_KINDS and isinstance(e.get("reason"), str) and e["reason"].strip()):
            raise ValueError(f"an exception is a path, a count above 0, a kind of {EXCEPT_KINDS} and a reason: {e!r}")
        if e["path"] in out:
            raise ValueError(f"{e['path']} is excepted twice")
        out[e["path"]] = e["count"]
    return out


def no_public_address(root: Path, finding: Mapping) -> tuple[bool, str]:
    allowed = excepted(finding)
    found = addresses.scan(addresses.tracked(root), addresses.reader(root))
    holding = sum(sum(kind == "ipv4" for kind, _ in hits) > allowed.get(f, 0) for f, hits in found.items())
    return holding == 0, f"{holding} tracked files hold a public IPv4 address the list doesn't except"


def outside_checkout(command: str, names: set[str]) -> bool:
    """Whether a hook's shell `command` runs the guard from outside the checkout: it splits into plain words (nothing the
    shell expands, joins or redirects), the first a file directly under `GUARD_ROOT`, and no other word is a path, `.` or names
    one of the checkout's top-level `names`. Anything else, a command that won't split included, counts as inside."""
    try:
        first, *rest = shlex.split(command)
    except ValueError:
        return False
    if not _GUARD_FILE.fullmatch(first):
        return False
    return all(_PLAIN.fullmatch(w) and ".." not in w and not set(re.split(r"[=:,]", w)) & (names | {"."}) for w in rest)


def guard_outside_checkout(root: Path, finding: Mapping) -> tuple[bool, str]:
    shown = subprocess.run(["git", "show", f"HEAD:{HOOKS}"], cwd=root, capture_output=True, text=True)
    if shown.returncode:
        return False, f"the commit tracks no {HOOKS}, so the tree declares no hook: a file a VM writes is not the tree's"
    hooks = json.loads(shown.stdout).get("hooks", {})
    commands = [h["command"] for entries in hooks.values() for h in entries]
    names = {p.split("/", 1)[0] for p in addresses.tracked(root)}
    inside = sum(not outside_checkout(c, names) for c in commands)
    return bool(commands) and inside == 0, (f"{inside} of {len(commands)} hook commands of {HOOKS} at the commit don't run "
                                            f"the guard from {GUARD_ROOT} alone")


#: each finding's check, by its id: the list names it, and can't hand a finding another's
CHECKS: Mapping[str, Check] = {
    "guard-runs-from-the-branch": Check("guard-outside-checkout", "declared", guard_outside_checkout),
    "machine-addresses-in-the-tree": Check("no-public-address", "tree", no_public_address),
}


def status(finding: Mapping, root: Path = ROOT, checks: Mapping[str, Check] = CHECKS) -> Status:
    """Whether `finding`, one [[finding]] of the list, is open, and what decided it."""
    fid = str(finding.get("id", "?"))
    named = finding.get("check")
    if (check := checks.get(fid)) is None:
        if named is not None:
            return Status(fid, True, "refused", f"it names the check {named!r}, and open_findings.py runs none for it")
        return Status(fid, True, "none", f"no check; what would close it: {finding.get('would_close', '')}")
    if (named, finding.get("reads")) != (check.name, check.reads):
        return Status(fid, True, "refused", f"it names the check {named!r} on {finding.get('reads')!r}, and "
                                            f"open_findings.py runs {check.name} on {check.reads!r}")
    try:
        closed, why = check.run(root, finding)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as e:
        return Status(fid, True, "refused", f"its check {check.name} couldn't run: {type(e).__name__}: {e}")
    return Status(fid, not closed, check.reads, why)


def mark(statuses: list[Status]) -> str:
    return "measured" if statuses and all(s.basis == "tree" for s in statuses) else "partial"


def load(path: Path = LIST) -> list[dict]:
    """The list's findings, each an id, a statement, and either a check (`check`, `reads`, `how`) or `would_close`: the
    check that would close it, in one line."""
    findings = tomllib.loads(path.read_text(encoding="utf-8")).get("finding", [])
    def words(f: Mapping, k: str) -> bool:
        return isinstance(f.get(k), str) and bool(f[k].strip())

    for f in findings:
        if not (words(f, "id") and words(f, "statement")):
            raise ValueError(f"{path}: each [[finding]] has an id and a statement: {f.get('id')!r}")
        if extra := sorted(set(f) - KEYS):
            raise ValueError(f"{path}: {f['id']} holds {', '.join(extra)}, which no finding holds: only a check closes one")
        if {"check", "reads", "how"} & set(f):
            whole = all(words(f, k) for k in ("check", "reads", "how")) and "would_close" not in f
        else:
            whole = words(f, "would_close") and "\n" not in f["would_close"]
        if not whole:
            raise ValueError(f"{path}: {f['id']} has either a check (check, reads and how) or, in one line, the check that "
                             f"would close it (would_close)")
    if len({f["id"] for f in findings}) != len(findings):
        raise ValueError(f"{path}: two [[finding]] entries share an id")
    return findings


def main(argv: Sequence[str] | None = None, path: Path = LIST, root: Path = ROOT,
         checks: Mapping[str, Check] = CHECKS) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args(argv).out
    statuses = [status(f, root, checks) for f in load(path)]
    n, m = sum(s.open for s in statuses), mark(statuses)
    print(f"security findings open: {n} of {len(statuses)} (mark {m})")
    for s in statuses:
        print(f"  {s.id}: {'open' if s.open else 'closed'} ({s.basis}): {s.why}")
    w = kit.writer(out)
    w.measurement(METRIC, n, "findings", target=0, mark=m, inputs=[str(path.relative_to(root))])
    refused = [s.id for s in statuses if s.basis == "refused"]
    w.validation("failed" if refused else "passed", "benchmarks/security_findings/open_findings.py",
                 f"entries refused: {', '.join(refused)}" if refused else f"read {len(statuses)} findings")


if __name__ == "__main__":
    main()
