"""`budgets.toml`: every spending cap in one file at the notes repository's root, read by the one guard (`research pods guard`) and
checked by `research pods create`.  A line is the approval: committing it replaces starting a guard process.

    [guard]
    project = ["vy"]                   # pod names starting with these are the project's; the guard sees no other pod
    balance_floor = 25                 # USD: at or under it, every project pod is terminated (the account's backstop)
    terminate_uncovered = true         # a project pod no line covers is terminated after uncovered_grace_min (false: reported only)
    uncovered_grace_min = 15
    exempt = ["vy-control-"]           # never counted, never terminated (the guard's own host is exempt without this)

    [budgets]
    "vy-circuit-checks-" = { cap_usd = 5, expires = "2026-09-30T06:00Z", max_pod_hours = 12, by = "daniel 2026-09-29" }
    "vy-coord-" = { cap_usd_per_day = 65, expires = "2026-10-31T00:00Z", max_pod_hours = 168, by = "daniel: the CI pool" }

A line covers every pod whose RunPod name starts with its prefix; when several do, the longest prefix covers it.
  cap_usd        the line's pods' integrated spend (costPerHr x time, from when the guard first sees the line) at or over it trips the
                 line: its pods are terminated, and `pods create` refuses under it until cap_usd is raised above the spend
  cap_usd_per_day  the same over any rolling 24 hours (not a calendar day, so no reset at midnight lets a day's spend double): its
                 trip clears by itself once the window's spend falls back under it.  A line needs cap_usd, cap_usd_per_day, or both
  expires        ISO 8601 with a zone: after it `pods create` refuses under the line; its pods are never terminated for the clock (their
                 leases end them)
  max_pod_hours  a pod under the line older than this (RunPod's createdAt) is terminated: the outer bound for a pod whose lease never
                 got armed; `pods create --max-hours` may not exceed it
  by             who approved it (free text)
A file that does not parse, or any line that does not validate, is rejected whole: the guard keeps the last good one, and says so.

Where it is read (`load`): a file path as given, or a notes clone, from its origin (`git fetch` then `git show origin/<branch>:budgets.toml`):
a line counts once it is pushed, whatever the clone's working tree holds.

`research pods create NAME ... --max-hours H` passes `gate` against the guard's own state, read from its host with one ssh call
($RESEARCH_GUARD_HOST, default GUARD_HOST; `local` reads the file here; the file is $RESEARCH_GUARD_STATE, default GUARD_STATE): it
refuses when the guard has not completed a poll in STALE_S, when its last read of budgets.toml failed (it enforces the last good
budgets, but nothing new starts on them), when the guard's budgets have no live, untripped line covering NAME, or when H is over the
line's max_pod_hours.  The guard's view is the one that counts: a line the guard has not read yet is not a line.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..notes_sync import remote_target

FILE = "budgets.toml"
GUARD_HOST = "vy-control-verity"
GUARD_STATE = "/root/.research/pods/guard-budgets.json"
STALE_S = 180
LINE_KEYS = {"cap_usd", "cap_usd_per_day", "expires", "max_pod_hours", "by"}
CAP_KEYS = ("cap_usd", "cap_usd_per_day")
GUARD_KEYS = {"project", "balance_floor", "terminate_uncovered", "uncovered_grace_min", "exempt"}


class BudgetError(ValueError):
    """budgets.toml does not parse or validate; the message lists every problem."""


def parse_time(s: Any) -> float | None:
    if isinstance(s, dt.datetime):
        return s.timestamp() if s.tzinfo else None
    try:
        d = dt.datetime.fromisoformat(str(s).strip())
    except ValueError:
        return None
    return d.timestamp() if d.tzinfo else None


def iso(t: float | None) -> str | None:
    return None if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Line:
    prefix: str
    cap_usd: float | None
    expires: float
    max_pod_hours: float
    by: str | None = None
    cap_usd_per_day: float | None = None

    def live(self, now: float) -> bool:
        return now < self.expires

    def view(self) -> dict[str, Any]:
        return {**asdict(self), "expires": iso(self.expires)}


@dataclass(frozen=True)
class Budgets:
    lines: dict[str, Line] = field(default_factory=dict)
    project: tuple[str, ...] = ()
    exempt: tuple[str, ...] = ()
    balance_floor: float | None = None
    terminate_uncovered: bool = True
    uncovered_grace_min: float = 15.0

    def covering(self, name: str) -> Line | None:
        hits = [ln for p, ln in self.lines.items() if name.startswith(p)]
        return max(hits, key=lambda ln: len(ln.prefix)) if hits else None

    def is_project(self, name: str) -> bool:
        return any(name.startswith(p) for p in self.project)

    def is_exempt(self, name: str) -> bool:
        return any(name.startswith(p) for p in self.exempt)

    def view(self) -> dict[str, Any]:
        return {"project": list(self.project), "exempt": list(self.exempt), "balance_floor": self.balance_floor,
                "terminate_uncovered": self.terminate_uncovered, "uncovered_grace_min": self.uncovered_grace_min,
                "lines": {p: ln.view() for p, ln in sorted(self.lines.items())}}


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse(text: str) -> Budgets:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise BudgetError(f"{FILE} does not parse: {e}") from None
    problems: list[str] = []
    for k in sorted(set(doc) - {"guard", "budgets"}):
        problems.append(f"unknown table or key {k!r} (only [guard] and [budgets])")
    g = doc.get("guard") or {}
    if not isinstance(g, dict):
        raise BudgetError(f"{FILE}: [guard] is not a table")
    for k in sorted(set(g) - GUARD_KEYS):
        problems.append(f"[guard] {k!r} is not a setting ({', '.join(sorted(GUARD_KEYS))})")
    project, exempt = g.get("project", []), g.get("exempt", [])
    for key, v in (("project", project), ("exempt", exempt)):
        if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            problems.append(f"[guard] {key} is not a list of non-empty name prefixes")
    floor = g.get("balance_floor")
    if floor is not None and not _num(floor):
        problems.append("[guard] balance_floor is not a number")
    grace = g.get("uncovered_grace_min", 15)
    if not _num(grace) or grace < 0:
        problems.append("[guard] uncovered_grace_min is not a number of minutes")
    tu = g.get("terminate_uncovered", True)
    if not isinstance(tu, bool):
        problems.append("[guard] terminate_uncovered is not true or false")
    raw = doc.get("budgets") or {}
    if not isinstance(raw, dict):
        raise BudgetError(f"{FILE}: [budgets] is not a table")
    lines: dict[str, Line] = {}
    for prefix, v in raw.items():
        where = f"[budgets] {prefix!r}"
        if not prefix.strip():
            problems.append(f"{where}: an empty prefix")
            continue
        if not isinstance(v, dict):
            problems.append(f"{where}: not an inline table {{ cap_usd = ..., expires = ..., max_pod_hours = ... }}")
            continue
        bad = sorted(set(v) - LINE_KEYS)
        missing = sorted({"expires", "max_pod_hours"} - set(v)) + ([] if any(k in v for k in CAP_KEYS) else ["cap_usd or cap_usd_per_day"])
        if bad:
            problems.append(f"{where}: {', '.join(bad)} not a line key ({', '.join(sorted(LINE_KEYS))})")
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
        if bad or missing:
            continue
        exp = parse_time(v["expires"])
        caps = [k for k in CAP_KEYS if k in v and (not _num(v[k]) or v[k] < 0)]
        if caps:
            problems.append(f"{where}: {' and '.join(caps)} {'is' if len(caps) == 1 else 'are'} not a number of dollars")
        elif exp is None:
            problems.append(f"{where}: expires {v['expires']!r} is not ISO 8601 with a zone (2026-09-30T06:00Z)")
        elif not _num(v["max_pod_hours"]) or v["max_pod_hours"] <= 0:
            problems.append(f"{where}: max_pod_hours is not a positive number of hours")
        else:
            lines[prefix] = Line(prefix, float(v["cap_usd"]) if "cap_usd" in v else None, exp, float(v["max_pod_hours"]),
                                 str(v["by"]) if v.get("by") is not None else None,
                                 float(v["cap_usd_per_day"]) if "cap_usd_per_day" in v else None)
    if problems:
        raise BudgetError(f"{FILE}: " + "; ".join(problems))
    return Budgets(lines=lines, project=tuple(project), exempt=tuple(exempt), balance_floor=float(floor) if floor is not None else None,
                   terminate_uncovered=tu, uncovered_grace_min=float(grace))


def _git(root: Path, *args: str, timeout: float = 60, opts: list[str] | None = None) -> str:
    r = subprocess.run(["git", "-C", str(root), *(opts or []), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise OSError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def read(source: Path) -> tuple[str, str]:
    """(text, provenance) of budgets.toml at `source`: a file, or a notes clone read from its origin after a fetch."""
    if not source.is_dir():
        return source.read_text(), str(source)
    branch = _git(source, "symbolic-ref", "--short", "HEAD").strip()
    ref = f"refs/remotes/origin/{branch}"
    opts, where = remote_target(source)
    _git(source, "fetch", "-q", where, f"+refs/heads/{branch}:{ref}", timeout=120, opts=opts)
    sha = _git(source, "rev-parse", ref).strip()
    return _git(source, "show", f"{ref}:{FILE}"), f"{source} origin/{branch} {sha[:12]}"


def load(source: Path) -> tuple[Budgets, str]:
    """Parsed budgets and where they came from; OSError / BudgetError when they cannot be had."""
    text, where = read(source)
    return parse(text), where


def default_source() -> Path:
    return Path(os.environ.get("RESEARCH_NOTES") or Path.home() / ".research" / "notes").expanduser()


# ---- the create gate ---------------------------------------------------------------------------------------------------------------------

def read_guard_state(host: str | None = None, path: str | None = None) -> tuple[dict[str, Any], float]:
    """(the guard's state, the host's clock) from the guard's host: one ssh call, `cat STATE; date +%s`.  `local` reads it here."""
    host = host or os.environ.get("RESEARCH_GUARD_HOST") or GUARD_HOST
    path = path or os.environ.get("RESEARCH_GUARD_STATE") or GUARD_STATE
    if host == "local":
        import time

        return json.loads(Path(path).read_text()), time.time()
    from .connect import ssh_base

    r = subprocess.run([*ssh_base(host), f"cat {path} && echo && date +%s"], capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        raise OSError(f"reading {path} on {host}: {(r.stderr or r.stdout).strip()[-300:] or f'rc {r.returncode}'}")
    body, _, clock = r.stdout.rstrip().rpartition("\n")
    return json.loads(body), float(clock)


def gate(name: str, max_hours: float, state: dict[str, Any], now: float, *, host: str = GUARD_HOST) -> str | None:
    """Why `pods create` must not create pod `name` with a `max_hours` lease, judged on the guard's state at `now`; None when it may."""
    last = state.get("last_ok_poll")
    if last is None or now - float(last) > STALE_S:
        ago = "never" if last is None else f"{(now - float(last)) / 60:.1f} min ago"
        return (f"the budgets guard on {host} last completed a poll {ago} (limit {STALE_S // 60} min): it is down or failing, and nothing "
                f"new starts unguarded (`research pods guard status` there)")
    b = state.get("budgets") or {}
    if b.get("error"):
        return (f"the budgets guard on {host} cannot read budgets.toml ({b['error']}); its last good read is {b.get('read_utc') or 'none'} "
                f"from {b.get('source') or '?'}, and nothing new starts on budgets it can't refresh: fix that read on {host} (a notes clone "
                f"whose HEAD is on its branch and whose origin it can fetch), then retry once a poll succeeds")
    lines = b.get("lines") or {}
    hits = [p for p in lines if name.startswith(p)]
    if not hits:
        return (f"no budget line covers {name!r} in the guard's budgets ({b.get('source') or '?'}). Commit one to budgets.toml in the "
                f"notes repo and push it, e.g. '\"{name.split('-veritor-campaign')[0]}-\" = {{ cap_usd = 5, expires = \"{iso(now + 86400)}\", "
                f"max_pod_hours = 12, by = \"...\" }}', then retry once the guard has read it (it re-reads every poll)")
    p = max(hits, key=len)
    ln, tally = lines[p], (state.get("lines") or {}).get(p) or {}
    exp = parse_time(ln.get("expires"))
    if exp is None or now >= exp:
        return f"the budget line {p!r} expired at {ln.get('expires')}: extend `expires` in budgets.toml to create under it"
    spent, day = float(tally.get("spent") or 0.0), float(tally.get("window_usd") or 0.0)
    cap, per_day = ln.get("cap_usd"), ln.get("cap_usd_per_day")
    if per_day is not None and (day >= float(per_day) or (tally.get("tripped") or {}).get("kind") == "per_day"):
        return (f"the budget line {p!r} is at its daily cap (${day:.2f} in the last 24 h of ${float(per_day):g}): it clears by itself as the "
                f"window moves, or raise cap_usd_per_day in budgets.toml")
    if tally.get("tripped") or (cap is not None and spent >= float(cap)):
        return (f"the budget line {p!r} is at its cap (${spent:.2f} of ${float(cap or 0):g}): raise cap_usd in budgets.toml to create "
                f"under it")
    if max_hours > float(ln["max_pod_hours"]):
        return f"--max-hours {max_hours:g} is over the line {p!r}'s max_pod_hours {float(ln['max_pod_hours']):g}"
    return None


def check_create(name: str, max_hours: float, *, reader=read_guard_state) -> tuple[str | None, str | None]:
    """(refusal, covering prefix) for `pods create`: the guard's state read from its host, judged by `gate`."""
    host = os.environ.get("RESEARCH_GUARD_HOST") or GUARD_HOST
    try:
        state, now = reader()
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        return (f"cannot read the budgets guard's state on {host} ({type(e).__name__}: {e}); nothing new starts unguarded", None)
    except Exception as e:  # noqa: BLE001  (connect.ssh_base refusals: no such pod, API down)
        return (f"cannot reach the budgets guard's host {host} ({type(e).__name__}: {e}); nothing new starts unguarded", None)
    why = gate(name, max_hours, state, now, host=host)
    lines = (state.get("budgets") or {}).get("lines") or {}
    hits = [p for p in lines if name.startswith(p)]
    return why, (max(hits, key=len) if hits else None)
