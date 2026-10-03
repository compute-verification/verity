"""The default rules of `research retention gc`: what the resource steward's sweeps deleted without asking (§1 of its policy,
note:20260930T2205Z-report-resource-steward), as implicit records, each with a rule name, an owner and an age.

    rule           owner    age   nodes        what
    src-tree       @infra   6 h   every        research/src/<sha>, aged from its mtime (when it was shipped), that holds nothing
                                               outside its commit (liveness.outside_commit); `research run` ships it again
    audit-tree     @proofs  6 h   vy-nebius-1  a source tree whose only change is the soundness lean-audit.json (LA, proofs' audit
                                               trees): the pass saves every such file in one evidence/v1 artifact, pushed
                                               preserved, before any of those trees goes
    lean-deps      @infra   any   vy-nebius-1  the .lake/packages of the verifier's three Lake packages (LAKE) in a source tree
                                               that stays: a failed audit leaves them, and the next audit takes them again
    jobs-src       @infra   6 h   vy-nebius-1  a job tree in jobs/src that no live pod reads (liveness.JobTrees)
    check-scratch  @infra   2 h   every        /tmp/pytest-of-research/pytest-<n> and <check cache>/lean-audit-scratch-* in which
                                               nothing changed for 2 h
    trash          @infra   any   every        what an interrupted pass renamed aside (src/.trash, jobs/src/.trash,
                                               .lake/packages.rs-trash-*)

Nothing live may name a path a rule deletes (liveness.Live: a source tree by its sha, anything else by its path). A source tree, a
job tree or a .lake/packages is renamed aside first (one rename, so nothing sees it half deleted); then every renamed path is checked
once more, put back if anything names it now (`stuck` when it can't be), and only then deleted. Scratch and trash are deleted where
they are. A rule never deletes PROTECTED paths (pouw/gpu3-fp8, check's verdict caches lean-deps and circuit-check), a path a record
covers (an explicit `research keep` wins), or one holding a record or a mount. `Sweep.run` raises Refused when what live work names
can't be established, or when node 2's guard (`timed`, read again before every deletion) says a timed window may run, after
putting back whatever is still renamed aside. Every decision about a path old enough for its rule goes to the deletion log (`Log`).
"""
from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Sequence

from .liveness import JobTrees, Live, LivenessError, Unknown, live_pods, outside_commit

HOUR = timedelta(hours=1)
NODE1 = frozenset({"vy-nebius-1"})
WORKSPACE = Path("/workspace")
CHECK_CACHE = Path("/home/research/.cache/verity-check")
PYTEST_TMP = Path("/tmp/pytest-of-research")
SHA = re.compile(r"[0-9a-f]{40}")
LAKE = ("backends/flock/verifier/lean", "backends/flock/verifier/lean/level3", "backends/flock/verifier/lean/soundness")
LA = "backends/flock/verifier/lean/soundness/lean-audit.json"
LOG_SCHEMA = "research/retention-log/v1"
LOCKS = (Path("/run/lock/vy-retention-gc.lock"), Path("/run/lock/resource-steward-sweep.lock"))
TS = "%Y-%m-%dT%H:%M:%SZ"
TIMED_NODES = frozenset({"vy-nebius-2"})
QUIET = "timed False, window waiting False"
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) ")
FRESH = timedelta(seconds=60)
SKEW_S = 2                               # a status written while `now` was read
LEAD = timedelta(minutes=15)


class Refused(RuntimeError):
    """gc may not act here now; the message names what unblocks it."""


class SaveError(RuntimeError):
    """The audit trees' lean-audit.json could not be preserved; the message says why."""


@dataclass(frozen=True)
class Rule:
    name: str
    owner: str
    age: timedelta
    nodes: frozenset[str] | None          # None: every node
    regenerate: str
    why: str


RULES = {r.name: r for r in (
    Rule("src-tree", "@infra", 6 * HOUR, None, "`research run` ships it again from the node's bare repo",
         "a shipped source tree that nothing live names and that holds nothing outside its commit"),
    Rule("audit-tree", "@proofs", 6 * HOUR, NODE1, "`research run` ships it again; its lean-audit.json is in the store",
         "proofs' audit tree (its only change is the soundness lean-audit.json), saved to the store before it goes"),
    Rule("lean-deps", "@infra", timedelta(0), NODE1, "check's next Lean audit takes them again",
         "the Lean dependencies a failed audit left in a source tree nothing live names"),
    Rule("jobs-src", "@infra", 6 * HOUR, NODE1, "job_tree.sh copies it again when a pod needs it", "a job tree no live pod reads"),
    Rule("check-scratch", "@infra", 2 * HOUR, None, "none needed: a check makes its own scratch",
         "check scratch in which nothing changed for 2 h"),
    Rule("trash", "@infra", timedelta(0), None, "none needed: an earlier pass decided it", "what an interrupted pass renamed aside"),
)}

DELETES = {"src-tree": "nothing live names its commit, and it holds nothing outside it",
           "lean-deps": "nothing live names its source tree",
           "jobs-src": "no live pod reads it, and nothing live names it",
           "check-scratch": "nothing under it changed for 2 h, and nothing live names it",
           "trash": "an interrupted pass left it, and nothing live names it"}


@dataclass(frozen=True)
class Places:
    """Where the rules look: research/, jobs/src and pouw/ under the root, and, for /workspace itself (`of`), the check scratch
    (/tmp/pytest-of-research, and the check cache that ~research/.cache/verity-check resolves to) and check's verdict caches."""
    root: Path
    scratch: tuple[tuple[Path, str], ...] = ()      # (a directory, the glob of the scratch in it)
    protected: tuple[Path, ...] = ()

    @property
    def research(self) -> Path:
        return self.root / "research"

    @property
    def src(self) -> Path:
        return self.research / "src"

    @property
    def bare(self) -> Path:
        return self.research / "git" / "verity.git"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs" / "src"

    @property
    def fill(self) -> Path:
        return self.root / "pouw" / "fill"

    @classmethod
    def of(cls, root: Path | str) -> Places:
        root = Path(os.path.abspath(root))
        protected = (root / "pouw" / "gpu3-fp8",)
        if root != WORKSPACE:
            return cls(root, (), protected)
        cache = Path(os.path.realpath(CHECK_CACHE))
        return cls(root, ((PYTEST_TMP, "pytest-[0-9]*"), (cache, "lean-audit-scratch-*")),
                   protected + (cache / "lean-deps", cache / "circuit-check"))


def timed(places: Places, node: str, now: datetime) -> str | None:
    """Why nothing may run on `node` now, else None. On node 2 (TIMED_NODES) the fill runner's pouw/fill/status.txt must exist,
    be written less than FRESH ago and stamped no later than now (the runner rewrites it every 10 s, so an older one is a stopped
    runner's), and say QUIET on its first line, as dispatch.py's n2_room reads it; and no booking in pouw/fill/windows may run now
    or start within LEAD (a booking without a GPU lease never shows in the status)."""
    if node not in TIMED_NODES:
        return None
    st, t = places.fill / "status.txt", now.timestamp()
    unsure = "so a timed window can't be ruled out: nothing runs on this node until the fill runner writes a fresh status"
    if not places.fill.is_dir():
        return f"{places.fill} is missing on {node}, {unsure}"
    try:
        written = os.stat(st).st_mtime
        with open(st, errors="replace") as f:
            first = f.readline().strip()
    except OSError as e:
        return f"{st} can't be read ({e.strerror or e}), {unsure}"
    if not (m := STAMP.match(first)):
        return f"{st}'s first line carries no time, {unsure}"
    fresh = FRESH.total_seconds()
    if not all(-SKEW_S <= t - x < fresh for x in (written, datetime.strptime(m[1], TS).replace(tzinfo=timezone.utc).timestamp())):
        return (f"{st} was written at {_fmt(datetime.fromtimestamp(written, timezone.utc))} and stamped {m[1]}, and it is "
                f"{_fmt(now)}: not both within the last {fresh:.0f} s, {unsure}")
    if QUIET not in first:
        return f"{st} does not say `{QUIET}` (a timed window runs or waits, or a cutover): nothing runs on this node until it does"
    try:
        bookings = booked(places.fill / "windows")
    except OSError as e:
        return f"{places.fill / 'windows'} can't be read ({e.strerror or e}), so a booking can't be ruled out"
    for start, end, line in bookings:
        if start is None:
            return f"{places.fill / 'windows'} has a line gc can't read ({line!r}), so a booking can't be ruled out: fix or remove it"
        if start - LEAD.total_seconds() <= t < end:
            return (f"{places.fill / 'windows'} books {line!r}: nothing runs on this node from {LEAD.total_seconds() / 60:.0f} min "
                    "before a booking until it ends")
    return None


def booked(path: Path) -> list[tuple[float | None, float, str]]:
    """(start, end, line) of each booking in a fill windows file, read as fill_runner.windows() reads it (a time without a zone is
    UTC; 30 min when no length is given); start None for a line that doesn't parse. OSError other than a missing file passes."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    out: list[tuple[float | None, float, str]] = []
    for line in lines:
        if not (f := line.split("#")[0].split()):
            continue
        try:
            start = datetime.fromisoformat(f[0])
            start = (start if start.tzinfo else start.replace(tzinfo=timezone.utc)).timestamp()
            out.append((start, start + 60 * float(f[1] if len(f) > 1 else 30), line.strip()))
        except ValueError:
            out.append((None, 0.0, line.strip()))
    return out


def lock(paths: Sequence[Path] = LOCKS) -> list[IO[str]]:
    """Hold every lock in `paths` (flock, as node-sweep.sh takes its own) while the returned files stay open; Refused when one is
    held or can't be taken."""
    held: list[IO[str]] = []
    try:
        for p in paths:
            try:
                f = open(p, "a")
            except OSError as e:
                raise Refused(f"can't open the lock {p} ({e.strerror or e}): gc --apply runs as root") from None
            held.append(f)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise Refused(f"{p} is held: another gc or the steward's sweep is running; run again once it ends") from None
    except Refused:
        for f in held:
            f.close()
        raise
    return held


def _fmt(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime(TS)


@dataclass
class Log:
    """The node's one deletion log: a JSON line (LOG_SCHEMA) per decision, appended as it is made."""
    path: Path
    node: str
    by: str
    now: Callable[[], datetime]

    def check(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a"):
                pass
        except OSError as e:
            raise Refused(f"can't write the deletion log {self.path} ({e.strerror or e}), so nothing would record a deletion") from None

    def __call__(self, action: str, path: Path | str, *, rule: str | None, owner: str | None, files: int | None = None,
                 bytes: int | None = None, age_h: float | None = None, reason: str = "") -> None:
        line = {"schema": LOG_SCHEMA, "t": _fmt(self.now()), "node": self.node, "by": self.by, "action": action, "path": str(path),
                "rule": rule, "owner": owner, "files": files, "bytes": bytes, "age_h": age_h, "reason": reason}
        with open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")


def inventory(path: Path) -> tuple[int, int, str | None]:
    """(files, apparent bytes, why it may not be deleted whole: a record or a mount beneath it), never following a symlink;
    OSError when it can't be read."""
    from .retention import SIDECAR

    if os.path.ismount(path):
        return 0, 0, f"{path} is a mount point"
    top = os.lstat(path).st_dev
    files = size = 0
    stack = [os.fspath(path)]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                st = e.stat(follow_symlinks=False)
                if e.is_dir(follow_symlinks=False):
                    if st.st_dev != top:
                        return files, size, f"{e.path} is a mount point"
                    stack.append(e.path)
                    continue
                files, size = files + 1, size + st.st_size
                if e.name.endswith(SIDECAR):
                    return files, size, f"{e.path} is a record, and it decides what beneath it goes"
    return files, size, None


def newest(path: Path) -> float:
    """The newest mtime of `path` and of everything beneath it on its filesystem; infinity when any of it can't be read."""
    try:
        top = os.lstat(path)
        t, stack = top.st_mtime, [os.fspath(path)]
        while stack:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    st = e.stat(follow_symlinks=False)
                    t = max(t, st.st_mtime)
                    if e.is_dir(follow_symlinks=False) and st.st_dev == top.st_dev:
                        stack.append(e.path)
    except OSError:
        return float("inf")
    return t


def _is_dir(p: Path) -> bool:
    return p.is_dir() and not p.is_symlink()


def _dirs(d: Path) -> list[Path]:
    try:
        return sorted(p for p in d.iterdir() if _is_dir(p))
    except OSError:
        return []


def job_copy(name: str) -> bool:
    return name != "by-pod" and not name.startswith(".") and ".partial." not in name


class RuleSet:
    """Which rule's implicit record a path has on node `node` (`retention.record_at`), and the time its age counts from."""

    def __init__(self, places: Places, node: str):
        self.places, self.node = places, node
        self._jobs: JobTrees | None = None
        self._parents = {places.src: "src-tree", places.src / ".trash": "trash", places.jobs / ".trash": "trash"}
        if self.applies("jobs-src"):
            self._parents[places.jobs] = "jobs-src"
        for base, _glob in places.scratch:
            self._parents.setdefault(base, "check-scratch")

    def applies(self, rule: str) -> bool:
        nodes = RULES[rule].nodes
        return nodes is None or self.node in nodes

    def match(self, path: Path) -> str | None:
        p = Path(os.path.abspath(path))
        rule = self._parents.get(p.parent)
        if rule is None or not _is_dir(p):
            return None
        if rule == "src-tree" and not SHA.fullmatch(p.name) or rule == "jobs-src" and not job_copy(p.name):
            return None
        if rule == "check-scratch" and not any(b == p.parent and fnmatch.fnmatchcase(p.name, g) for b, g in self.places.scratch):
            return None
        return rule

    def stamp(self, rule: str, path: Path) -> float:
        if rule == "check-scratch":
            return newest(path)
        if rule == "jobs-src":
            self._jobs = self._jobs or JobTrees(self.places.jobs)
            return self._jobs.age_stamp(Path(path).name)
        return os.lstat(path).st_mtime

    def outside(self) -> Iterable[tuple[Path, str]]:
        """(path, rule) of what a rule covers outside the root: the check scratch in /tmp or in a check cache elsewhere."""
        for base, g in self.places.scratch:
            if self.places.root in base.parents or base == self.places.root:
                continue
            for p in sorted(base.glob(g)) if base.is_dir() else []:
                if _is_dir(p):
                    yield p, "check-scratch"


@dataclass
class Act:
    path: Path
    rule: str
    action: str                  # "delete" | "keep" | "stuck"
    reason: str
    files: int | None = None
    bytes: int | None = None
    age_h: float | None = None
    error: str | None = None

    def doc(self) -> dict[str, Any]:
        return {"path": str(self.path), "rule": self.rule, "owner": RULES[self.rule].owner, "action": self.action, "reason": self.reason,
                "files": self.files, "bytes": self.bytes, "age_h": self.age_h, "error": self.error}


@dataclass
class _Cand:
    path: Path
    rule: str
    stamp: float
    files: int = 0
    bytes: int = 0
    patterns: tuple[str, ...] = ()       # what names it in the check after the rename
    trash: Path | None = None            # where it was renamed to
    reason: str = ""


def _refs(refs: list[str]) -> str:
    return ", ".join(refs[:3]) + (f" and {len(refs) - 3} more" if len(refs) > 3 else "")


@dataclass
class Sweep:
    """One pass of the rules over `places` on node `node` (module docstring). `covered(path)`: why a record keeps it, else None.
    `save(trees)`: the art id preserving their lean-audit.json, or SaveError. `log`: where an --apply's decisions go."""
    places: Places
    node: str
    now: Callable[[], datetime]
    live: Live
    apply: bool = False
    covered: Callable[[Path], str | None] = lambda _p: None
    save: Callable[[list[Path]], str] | None = None
    log: Log | None = None
    acts: list[Act] = field(default_factory=list)

    def run(self) -> list[Act]:
        self.t = self.now().timestamp()
        self.rules = rules = RuleSet(self.places, self.node)
        pl = self.places
        trash = [(p, "trash") for d in (pl.src / ".trash", pl.jobs / ".trash") for p in _dirs(d)]
        trash += [(p, "trash") for t in _dirs(pl.src) if SHA.fullmatch(t.name) for k in LAKE
                  for p in sorted((t / k / ".lake").glob("packages.rs-trash-*")) if _is_dir(p)]
        trees = [t for t in _dirs(pl.src) if SHA.fullmatch(t.name)]
        old = [t for t in trees if self._old("src-tree", os.lstat(t).st_mtime)]
        lake = {t: pk for t in trees if rules.applies("lean-deps")
                if (pk := [t / k / ".lake" / "packages" for k in LAKE if _is_dir(t / k / ".lake" / "packages")])}
        scratch = [(p, s) for base, g in pl.scratch for p in (sorted(base.glob(g)) if base.is_dir() else [])
                   if _is_dir(p) and self._old("check-scratch", s := newest(p))]
        jobs, jt = self._job_copies(rules)
        patterns = ([t.name for t in dict.fromkeys([*old, *lake])] + [str(p) for p, _ in trash] + [str(p) for p, _ in scratch]
                    + [str(pl.jobs / e) for e, _ in jobs])
        if not patterns:
            return self.acts
        named = self._names(patterns)

        going: list[_Cand] = []
        for p, rule in trash:
            if c := self._vet(p, rule, os.lstat(p).st_mtime, named.get(str(p), [])):
                going.append(c)
        audit: list[_Cand] = []
        for t in old:
            if c := self._vet(t, "src-tree", os.lstat(t).st_mtime, named.get(t.name, []), (t.name,), tree=True):
                (audit if c.rule == "audit-tree" else going).append(c)
        if audit and self.apply:
            audit = self._save(audit)
        going += audit
        whole = {c.path for c in going if c.rule in ("src-tree", "audit-tree")}
        for t, pks in lake.items():
            if t in whole:
                continue
            for pk in pks:
                if c := self._vet(pk, "lean-deps", os.lstat(t).st_mtime, named.get(t.name, []), (t.name,)):
                    going.append(c)
        for p, s in scratch:
            if c := self._vet(p, "check-scratch", s, named.get(str(p), []), (str(p),)):
                going.append(c)
        for e, s in jobs:
            p = pl.jobs / e
            if c := self._vet(p, "jobs-src", s, named.get(str(p), []), (str(p), str(pl.jobs / ".trash" / e))):
                going.append(c)

        if not self.apply:
            for c in going:
                why = (c.reason if c.rule == "audit-tree" else DELETES[c.rule])
                self._add(Act(c.path, c.rule, "delete", why, c.files, c.bytes, self._age(c.stamp)))
            return self.acts
        self._pending: list[_Cand] = []
        try:
            for c in [c for c in going if c.rule == "trash"]:
                self._quiet()
                self._remove(c, c.path)
            self._quiet()
            self._rename([c for c in going if c.rule in ("src-tree", "audit-tree", "lean-deps", "jobs-src")])
            self._recheck([c for c in going if c.rule == "check-scratch"], jt is not None)
        except Refused as e:
            self._undo(f"put back undecided: {e}")
            raise
        return self.acts

    # ---- the steps --------------------------------------------------------------------------------------------------------------------

    def _old(self, rule: str, stamp: float) -> bool:
        return self.t - stamp >= RULES[rule].age.total_seconds()

    def _age(self, stamp: float) -> float:
        return round((self.t - stamp) / 3600, 1)

    def _names(self, patterns: list[str]) -> dict[str, list[str]]:
        try:
            return self.live.names(patterns)
        except LivenessError as e:
            raise Refused(f"can't establish what live work names ({e}): nothing deleted") from None

    def _job_copies(self, rules: RuleSet) -> tuple[list[tuple[str, float]], JobTrees | None]:
        pl = self.places
        if not rules.applies("jobs-src") or not (pl.jobs / "by-pod").is_dir():
            return [], None
        if self.live.kube is None:
            raise Refused(f"can't tell which pods are live (no k3s here), and {pl.jobs}'s copies need it: nothing deleted")
        try:
            jt = JobTrees(pl.jobs, live_pods(self.live.kube))
        except LivenessError as e:
            raise Refused(f"can't establish which pods are live ({e}): nothing deleted") from None
        out = []
        for e, why in jt.copies():
            stamp = jt.age_stamp(e)
            if why:
                self._add(Act(pl.jobs / e, "jobs-src", "keep", why, age_h=self._age(stamp)))
            elif self._old("jobs-src", stamp):
                if held := jt.held(e):
                    self._add(Act(pl.jobs / e, "jobs-src", "keep", held, age_h=self._age(stamp)))
                else:
                    out.append((e, stamp))
        return out, jt

    def _vet(self, p: Path, rule: str, stamp: float, refs: list[str], patterns: tuple[str, ...] = (), tree: bool = False) -> _Cand | None:
        """The candidate `p` is when nothing keeps it (`patterns`: what names it in the check after the rename); else None after its
        keep."""
        def keep(why: str) -> None:
            self._add(Act(p, rule, "keep", why, age_h=self._age(stamp)))

        if any(p == q or q in p.parents or p in q.parents for q in self.places.protected):
            return keep("protected: no rule deletes it")
        if why := self.covered(p):
            return keep(why)
        if refs:
            return keep(f"named by {_refs(refs)}")
        reason = ""
        if tree:
            try:
                extra = outside_commit(p, self.places.bare)
            except Unknown as e:
                return keep(f"can't tell what it holds besides its commit: {e}")
            if extra == [f"M {LA}"] and self.rules.applies("audit-tree"):
                rule, reason = "audit-tree", "would save its lean-audit.json to the store, then delete it"
            elif extra:
                return keep(f"{len(extra)} file(s) outside its commit, e.g. {', '.join(extra[:3])}: they may be a run's output, so "
                            "its owner decides")
        try:
            files, size, why = inventory(p)
        except OSError as e:
            return keep(f"can't read it: {e.strerror or e}")
        if why:
            return keep(why)
        return _Cand(p, rule, stamp, files, size, patterns, reason=reason)

    def _save(self, audit: list[_Cand]) -> list[_Cand]:
        try:
            if self.save is None:
                raise SaveError("nothing here saves it")
            art = self.save([c.path for c in audit])
        except SaveError as e:
            for c in audit:
                self._add(Act(c.path, c.rule, "keep", f"its lean-audit.json is not saved, so it stays: {e}", age_h=self._age(c.stamp)))
            return []
        for c in audit:
            c.reason = f"its lean-audit.json is saved as {art}, and nothing live names its commit"
        return audit

    def _quiet(self) -> None:
        if why := timed(self.places, self.node, self.now()):
            raise Refused(f"{why} (seen during the pass: what is renamed aside goes back, and nothing more is deleted)")

    def _rename(self, cands: list[_Cand]) -> None:
        pl, stamp = self.places, int(self.now().timestamp())
        self.renamed_at = float(stamp)
        for c in cands:
            c.trash = (c.path.with_name(f"packages.rs-trash-{stamp}") if c.rule == "lean-deps" else
                       pl.jobs / ".trash" / c.path.name if c.rule == "jobs-src" else pl.src / ".trash" / f"{c.path.name}-{stamp}")
            self._pending.append(c)
            try:
                c.trash.parent.mkdir(exist_ok=True)
                os.rename(c.path, c.trash)
            except OSError as e:
                self._pending.remove(c)
                self._add(Act(c.path, c.rule, "keep", f"the rename aside failed: {e.strerror or e}", age_h=self._age(c.stamp)))

    def _recheck(self, scratch: list[_Cand], jobs: bool) -> None:
        moved = list(self._pending)
        if not moved and not scratch:
            return
        try:
            named = self.live.names([p for c in moved for p in c.patterns] + [str(c.path) for c in scratch])
            jt = JobTrees(self.places.jobs, live_pods(self.live.kube)) if jobs and any(c.rule == "jobs-src" for c in moved) else None
        except LivenessError as e:
            raise Refused(f"can't establish what live work names after the renames ({e}): what was renamed is put back, and "
                          "nothing else deleted") from None
        for c in moved:
            refs = [r for p in c.patterns for r in named.get(p, [])]
            if c.rule == "jobs-src" and jt is not None and (held := jt.held(c.path.name, since=self.renamed_at)):
                refs.append(held)
            if refs:
                self._put_back(c, f"named after the rename: {_refs(refs)}")
            else:
                self._quiet()
                self._remove(c, c.trash)
        for c in scratch:
            if refs := named.get(str(c.path), []):
                self._add(Act(c.path, c.rule, "keep", f"named by {_refs(refs)}", age_h=self._age(c.stamp)))
            else:
                self._quiet()
                self._remove(c, c.path)

    def _undo(self, why: str) -> None:
        """Put back every candidate still renamed aside (one appended just before its rename may not be)."""
        for c in list(self._pending):
            if c.trash is not None and os.path.lexists(c.trash):
                self._put_back(c, why)
            else:
                self._pending.remove(c)

    def _put_back(self, c: _Cand, why: str) -> None:
        self._pending.remove(c)
        try:
            if os.path.lexists(c.path):
                raise FileExistsError(f"a new {c.path} is there")
            os.rename(c.trash, c.path)
        except OSError as e:
            self._add(Act(c.path, c.rule, "stuck", f"{why}, and it can't be put back from {c.trash}: {e}", c.files, c.bytes,
                          self._age(c.stamp)))
            return
        self._add(Act(c.path, c.rule, "keep", why, age_h=self._age(c.stamp)))

    def _remove(self, c: _Cand, where: Path) -> None:
        if c in self._pending:
            self._pending.remove(c)
        try:
            shutil.rmtree(where)
            err = None
        except OSError as e:
            err = f"{type(e).__name__}: {e}"
        self._add(Act(c.path, c.rule, "delete", c.reason or DELETES[c.rule], c.files, c.bytes, self._age(c.stamp), err))

    def _add(self, a: Act) -> None:
        self.acts.append(a)
        if self.apply and self.log is not None:
            action = {"keep": "kept", "stuck": "stuck"}.get(a.action) or ("failed" if a.error else "deleted")
            self.log(action, a.path, rule=a.rule, owner=RULES[a.rule].owner, files=a.files, bytes=a.bytes, age_h=a.age_h,
                     reason=a.reason + (f" ({a.error})" if a.error else ""))


def store_saver(store_root: str | None, node: str, by: str | None) -> Callable[[list[Path]], str]:
    """The default `save`: the trees' lean-audit.json as one evidence/v1 artifact (a tree of <sha>/LA), pushed preserved; its art id,
    else SaveError. Nothing is put where no remote is usable, since a local copy alone preserves nothing."""
    def save(trees: list[Path]) -> str:
        from .store.local import LocalStore, StoreError
        from .store.remote import ConfigError, load_remote

        try:
            if load_remote() is None:
                raise SaveError("no store remote is configured here (store.toml or R2_*), so a copy here would preserve nothing")
        except ConfigError as e:
            raise SaveError(f"the store's remote is unusable here: {e}") from None
        meta = {"lane": "proofs", "what": "the modified soundness lean-audit.json of source trees retention gc deletes",
                "layout": f"<sha>/{LA}", "node": node, "trees": [t.name for t in trees], **({"by": by} if by else {})}
        with tempfile.TemporaryDirectory(prefix="retention-lean-audit-") as d:
            for t in trees:
                (Path(d) / t.name / LA).parent.mkdir(parents=True)
                shutil.copy2(t / LA, Path(d) / t.name / LA)
            try:
                st = LocalStore(store_root)
                art = st.put_artifact("evidence/v1", meta, tree=Path(d))
                rep = st.push(art, preserve=True)
            except (StoreError, OSError) as e:
                raise SaveError(str(e)) from None
        if not rep.get("preserved"):
            raise SaveError(f"{art} is not preserved: {(rep.get('errors') or ['?'])[0]}")
        return art

    return save


__all__ = ["Act", "FRESH", "LA", "LAKE", "LEAD", "LOCKS", "LOG_SCHEMA", "Log", "NODE1", "Places", "QUIET", "RULES", "Refused", "Rule",
           "RuleSet", "SaveError", "Sweep", "TIMED_NODES", "booked", "inventory", "lock", "newest", "store_saver", "timed"]
