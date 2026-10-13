"""`research queue drive`: one tick of the landing automation (Daniel, 11 Oct: from an owner's ready mark to `main` moving, no
agent acts unless the automation refuses).  A loop runs it every few minutes; each tick does at most one of each step:

1. Land.  Walk the queue's branch down from its tip to the newest commit with a passing `check` and land it as the lander does:
   a merge commit onto `main` naming its PRs, with a `Checked:` trailer for the run.  On a `main` that moved outside the queue,
   the commit still lands when the commits `main` gained share no file with it (`Skipped-check:` names them) and change
   no check tooling (`CHECK_TOOLING`), which would have checked it differently.  Then
   `merge.caught_up` (the spec DM, node 1's mirror, the post-train regression, the landing post) and the sweep.
2. Sync `next` (`Queue.sync`): admit and drop PRs, rebuild on a moved `main`; but while a check of `next` runs and `main` moved
   by files its train doesn't change (and no check tooling), `next` stays on its base so the check isn't wasted (`held`).
3. Check `next` in trains of `train_size` PRs, each stacked on the one below (every `train_size`-th member's commit, and the
   tip): `train.sh COMMIT MACHINE`, with `--agreement` when the train changes a `lean-audit.json` (the lander's rule; train.sh
   itself sends the build for C-Flock's paths), the lowest unchecked first, while fewer than `max_running` checks run.

What it refuses goes to `@ci` in the merge thread, once per commit or head, with the owner of the PR it concerns: a failed
check, a conflict, a check that ran and passed but lacks a step the gate needs, a launch that registered no run.  A failed
check whose failing tests' files and tracebacks only one PR of the train changes (`implicated`) ejects that PR itself
(`research queue eject`, by `ci`); otherwise `@ci` ejects the PR that broke it.  The next sync rebuilds `next` without it.

`--shadow` drives `queue-shadow`: it lands nothing, launches nothing and posts nothing, and logs what it would do.

State (launches, refusals told) is the queue's `state.json`, under `drive`."""
from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .queue import IDENTITY, Member, Queue, QueueError, _git, _out

#: who hears what drive can't settle (13 Oct: the other Project's lander, its addressee before, acts on none of it)
CI = "@ci"
#: a pytest location (`tests/test_x.py:503: NameError`) or a traceback frame (`File "/.../x.py", line 12`) in a suite's log
TRACE_RE = re.compile(r'^\s*([\w./+-]+\.py):\d+:|File "([^"]+\.py)", line \d+', re.M)
#: states `research status` reports for a run that may still publish an attempt
ACTIVE = ("submitted", "waiting", "running", "unknown")
#: after a launched run ends, how long its attempt may take to reach the store before the launch counts as lost
PUBLISH_GRACE_S = 1800
#: check.py's exit for a twin that never took a slot
NOT_STARTED = 75
#: what decides how a check runs and what it passes: a `main` that changed any of it re-checks the queue rather than land on it
CHECK_TOOLING = ("tools/verity/check/", "tools/verity/lean/", "tools/verity/circuit_check/")
RUN_RE = re.compile(r"^RUN ((?:r20\d{6}-\d{6}-[0-9a-f]{4} ?)+)$", re.M)


@dataclass
class Verdict:
    state: str                    # passed | running | failed | none
    attempt: str = ""
    why: str = ""


def changed(repo: Path, base: str, commit: str) -> set[str]:
    return set(_out(repo, "diff", "--name-only", "--no-renames", base, commit).split())


def needs_agreement(repo: Path, base: str, commit: str) -> bool:
    """The lander's rule: a train that adds or changes a guarantee anywhere is checked with the pinned upstream build."""
    return any(p == "lean-audit.json" or p.endswith("/lean-audit.json") for p in changed(repo, base, commit))


def auth_env(auth: tuple[str, ...]) -> dict[str, str]:
    """The queue's `git -c` options (`credential_args`) as GIT_CONFIG_* variables, after any the caller carries, so every git a
    child runs in the queue's clone authenticates as the queue's own does."""
    pairs = [o.split("=", 1) for flag, o in zip(auth[::2], auth[1::2]) if flag == "-c"]
    n = int(os.environ.get("GIT_CONFIG_COUNT") or 0)
    env = {"GIT_CONFIG_COUNT": str(n + len(pairs))}
    for i, (k, v) in enumerate(pairs, n):
        env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = k, v
    return env if pairs else {}


@contextlib.contextmanager
def authed(auth: tuple[str, ...]):
    """os.environ with the queue's credentials (`auth_env`) and git never prompting, for the steps after a landing
    (`merge.caught_up`, `_sweep_after`), whose git children read os.environ (12 Oct 08:49Z: `caught_up`'s `ls-remote` in the
    queue's clone waited on a password at the tty and held drive for 13 min)."""
    extra = {**auth_env(auth), "GIT_TERMINAL_PROMPT": "0"}
    was = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    try:
        yield
    finally:
        for k, v in was.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def train_launch(repo: Path, commit: str, machine: str, agreement: bool, auth: tuple[str, ...] = ()) -> tuple[list[str], str]:
    """`train.sh COMMIT MACHINE [--agreement]` from the queue's clone: (the run ids it printed, its output's tail).  Git never
    prompts (12 Oct: train.sh's fetch in the queue's clone, which has no credential helper, waited on a password at the tty)."""
    argv = ["bash", "tools/verity/check/train.sh", commit, machine, *(["--agreement"] if agreement else [])]
    r = subprocess.run(argv, cwd=repo, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       env={**os.environ, **auth_env(auth), "GIT_TERMINAL_PROMPT": "0", "TRAIN_AS": "ci"})
    m = RUN_RE.search(r.stdout)
    return (m[1].split() if m else []), (r.stdout + r.stderr).strip()[-600:]


#: what a running check has written of its steps and suites so far (check.py's `STEPS_LOG`, suites.py's `failing_path` of each
#: `--json`), fetched with its records
PROGRESS = ("steps.jsonl", "suites.failed.json", "suites-lean.failed.json")


def failing(run_dir: Path) -> str | None:
    """What has failed so far in a check still running under `--keep-going`, which fails it whatever else it finds, or None."""
    out = []
    try:
        lines = (run_dir / PROGRESS[0]).read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("status") == "failed":
            out.append(f"step {rec.get('step')}")
    for name in PROGRESS[1:]:
        try:
            doc = json.loads((run_dir / name).read_text())
        except (OSError, ValueError):
            continue
        for s in doc if isinstance(doc, list) else ():
            tests = s.get("failed_tests") or []
            out.append(f"suite {s.get('name')}" + (f" ({', '.join(tests[:2])}{' ...' if len(tests) > 2 else ''})" if tests else ""))
    return "; ".join(out) or None


def run_states(repo: Path, ids: list[str]) -> dict[str, tuple[str, int | None, str | None, str | None]]:
    """Each run's (state, rc, ended_utc, what has failed so far while it runs: `failing`) from `research status RUN --json`, which
    fetches `PROGRESS` with its records; a run it can't read is ("unknown", None, None, None)."""
    out = {}
    also = [x for n in PROGRESS for x in ("--also", n)]
    for rid in ids:
        r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--json", *also], cwd=repo, capture_output=True, text=True)
        try:
            run = json.loads(r.stdout)["runs"][0]
            state = run.get("state") or "unknown"
            out[rid] = (state, run.get("rc"), run.get("ended_utc"),
                        failing(Path(run["dir"])) if state in ACTIVE and run.get("dir") else None)
        except (ValueError, KeyError, IndexError):
            out[rid] = ("unknown", None, None, None)
    return out


def cancel_runs(repo: Path, ids: list[str], reason: str) -> None:
    for rid in ids:
        subprocess.run([sys.executable, "-m", "research", "cancel", rid, "--by", "ci", "--reason", reason], cwd=repo,
                       capture_output=True, text=True)


def merge_thread_post(text: str, to: tuple[str, ...] = (CI,)) -> bool:
    r = subprocess.run([sys.executable, "-m", "research", "msg", "send", "--as", "ci", "--thread", _merge_thread(), "--to", *to,
                        "--quiet", "--text", text], capture_output=True, text=True)
    return r.returncode == 0


def _merge_thread() -> str:
    from .monitors import MERGE_THREAD

    return MERGE_THREAD


def _utc(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return time.mktime(time.strptime(ts.rstrip("Z")[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except ValueError:
        return None



def _docs(run_dir: Path, *names: str) -> list[dict]:
    out = []
    for name in names:
        try:
            doc = json.loads((run_dir / name).read_text())
        except (OSError, ValueError):
            continue
        out += [x for x in doc if isinstance(x, dict)] if isinstance(doc, list) else []
    return out


def failed_suites(run_dir: Path) -> list[dict]:
    """Each failed suite's record: from `suites.json` and `suites-lean.json` once the check ended, else from what it has written
    of its failures so far (`PROGRESS`)."""
    done = [s for s in _docs(run_dir, "suites.json", "suites-lean.json") if s.get("status") == "failed"]
    return done or _docs(run_dir, *PROGRESS[1:])


def _fetch(repo: Path, rid: str, names: list[str]) -> Path | None:
    also = [x for n in names for x in ("--also", n)]
    r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--json", *also], cwd=repo, capture_output=True, text=True)
    try:
        return Path(json.loads(r.stdout)["runs"][0]["dir"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def evidence(repo: Path, ids: list[str]) -> list[Path]:
    """The local run dirs of ``ids`` with their steps, suite reports and failed suites' logs fetched (`research status --also`)."""
    dirs = []
    for rid in ids:
        if (d := _fetch(repo, rid, [*PROGRESS, "suites.json", "suites-lean.json"])) is None:
            continue
        if logs := [f"suites/{Path(s['log']).name}" for s in failed_suites(d) if s.get("log")]:
            _fetch(repo, rid, logs)
        dirs.append(d)
    return dirs


def implicated(run_dir: Path) -> set[str] | None:
    """The paths a failed check's failure points at: each failed suite's failing test files and the files its log's tracebacks
    name (relative to the repository, or absolute as the log wrote them); None when a step other than the suites failed, or a
    failed suite names no failing test (a timeout, a kill), so no file says what broke it."""
    steps = []
    with contextlib.suppress(OSError):
        for line in (run_dir / PROGRESS[0]).read_text().splitlines():
            with contextlib.suppress(ValueError):
                if (rec := json.loads(line)).get("status") == "failed":
                    steps.append(rec.get("step"))
    suites = failed_suites(run_dir)
    if not suites or any(st not in ("pytest", "lean-suites") for st in steps):
        return None
    paths: set[str] = set()
    for s in suites:
        base, tests = s.get("suite") or "", s.get("failed_tests") or []
        if not tests:
            return None
        paths |= {os.path.normpath(os.path.join(base, t.split("::")[0])) for t in tests}
        if s.get("log"):
            with contextlib.suppress(OSError):
                for m in TRACE_RE.finditer((run_dir / "suites" / Path(s["log"]).name).read_text(errors="replace")):
                    p = m[1] or m[2]
                    paths.add(p if p.startswith("/") else os.path.normpath(os.path.join(base, p)))
    return paths


def touches(files: set[str], paths: set[str]) -> set[str]:
    """The ``files`` (repository paths) among ``paths``, an absolute path matching the file it ends in when that file is under a
    directory (a bare `conftest.py` would match every conftest an installed package has)."""
    return {f for f in files for p in paths if p == f or ("/" in f and p.startswith("/") and p.endswith("/" + f))}


class Driver:
    def __init__(self, q: Queue, *, on: str, max_running: int = 2, train_size: int = 4,
                 launch: Callable[[Path, str, str, bool], tuple[list[str], str]] = train_launch,
                 states: Callable[[Path, list[str]], dict] = run_states, post: Callable[[str, tuple[str, ...]], bool] = merge_thread_post,
                 now: Callable[[], float] = time.time, cancel: Callable[[Path, list[str], str], None] = cancel_runs,
                 evidence: Callable[[Path, list[str]], list[Path]] = evidence):
        self.q, self.on, self.max_running, self.train_size = q, on, max_running, train_size
        if launch is train_launch:
            launch = functools.partial(train_launch, auth=q.auth)
        self.launch, self.states, self.post, self.now, self.cancel, self.evidence = launch, states, post, now, cancel, evidence
        self.repo, self.dry_run, self._seen, self._prs = q.workdir, False, {}, None

    # -- what the store says of a commit

    def _states(self, ids: list[str]) -> dict:
        """`states` of ``ids``, each run read once a tick: one `research status` is a few seconds over ssh, and a tick asks of the
        same launch for `held`, before the sync, for the running count and for its train (12 Oct: 25 s of a 52 s tick)."""
        if missing := [rid for rid in ids if rid not in self._seen]:
            self._seen.update(self.states(self.repo, missing))
        return {rid: self._seen[rid] for rid in ids}

    def verdict(self, commit: str, base: str, drive: dict) -> Verdict:
        """`commit`'s check: passed (the gate's rule, against `base`), failed, running (a launch of ours not yet in the store),
        or none (never launched, or every twin of the last launch exited without starting)."""
        from .commitmap import Maps
        from .merge import declared_command, passing, produced, refused, required_steps, step_status

        store = self.q.store
        tree = _out(self.repo, "rev-parse", f"{commit}^{{tree}}")
        kinds, command = produced(self.repo, "check"), declared_command(self.repo, "check")
        ids = passing(store, commit, "check", kinds=kinds, command=command, tree=tree, maps=Maps.load(store), repo=self.repo)
        need = required_steps(self.repo, base, commit, "check")
        for rid in ids:
            if all(step_status(store, rid, s) == "passed" for s in need):
                return Verdict("passed", rid)
        rows = store.index.sql("SELECT id, state, rc FROM attempts WHERE tool = 'check' AND source_commit = ?", (commit,))
        ran = [(rid, st, rc) for rid, st, rc in rows if rc != NOT_STARTED]
        if ids:
            return Verdict("failed", ids[0], f"{ids[0]} passed without {', '.join(need)}, which the gate needs for "
                                            f"{', '.join(sorted({p for ps in need.values() for p in ps}))}")
        if ran:
            rid, st, rc = ran[-1]
            why = refused(store, rid, kinds, "check", commit, command, self.repo) if st == "done" and rc == 0 else f"rc={rc}"
            return Verdict("failed", rid, f"{rid} {st} {why}")
        launch = drive["launched"].get(commit)
        if not launch:
            return Verdict("none")
        states = self._states(launch["ids"])
        if early := next((s[3] for s in states.values() if len(s) > 3 and s[0] in ACTIVE and s[3]), None):
            launch["failing"] = early
            return Verdict("failed", " ".join(launch["ids"]), f"{' '.join(launch['ids'])} is still running (`--keep-going`, for "
                                                              f"the full picture) and has already failed: {early}")
        if any(st in ACTIVE for st, *_ in states.values()):
            return Verdict("running", " ".join(launch["ids"]))
        if all(rc == NOT_STARTED for _, rc, *_ in states.values()):
            return Verdict("none", why="every twin exited without starting")
        ended = max((_utc(e) or launch["at"] for _, _, e, *_ in states.values()), default=launch["at"])
        if self.now() - ended < PUBLISH_GRACE_S:
            return Verdict("running", " ".join(launch["ids"]), "ended; its attempt isn't in the store yet")
        return Verdict("failed", " ".join(launch["ids"]),
                       f"{' '.join(launch['ids'])} ended ({', '.join(f'{s} rc={rc}' for s, rc, *_ in states.values())}) "
                       f"and no attempt reached the store in {PUBLISH_GRACE_S // 60} min")

    # -- landing

    def _chain(self) -> tuple[str, list[Member]]:
        """(the base `next` was built on, its members oldest first), whether or not `main` has moved since."""
        ref = f"refs/remotes/origin/{self.q.branch}"
        if _git(self.repo, "rev-parse", "--verify", "-q", ref, check=False).returncode:
            return "", []
        # not `merge-base`: a member stacked on a PR that has since landed on `main` gives a second, wrong common ancestor
        # (12 Oct 10:53Z: #1847 on #1793, after the e2e train's merge commit, emptied the chain and cancelled a disjoint check)
        base = next((c for c in _out(self.repo, "rev-list", "--first-parent", ref).split()
                     if not _git(self.repo, "merge-base", "--is-ancestor", c, "origin/main", check=False).returncode), "")
        if not base:
            return "", []
        from .queue import TRAILER_RE

        out = []
        for rec in filter(None, (x.strip() for x in _out(self.repo, "log", "--first-parent", "--reverse",
                                                          "--format=%H%x00%B%x1e", f"{base}..{ref}").split("\x1e"))):
            commit, body = rec.split("\x00", 1)
            t = dict(TRAILER_RE.findall(body))
            if not {"PR", "Head", "Owner"} <= set(t):
                return base, []
            out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
        return base, out

    def held(self, drive: dict) -> str | None:
        """Why `next` stays on its base this tick, or None: a check of it is running and `main` moved by files none of the running
        train's members change, and by no check tooling.  A sync would rebuild `next` on the new `main` and waste that check;
        `land` merges the train onto the moved `main` once it passes."""
        base, chain = self._chain()
        main = _out(self.repo, "rev-parse", "origin/main")
        if not chain or main == base:
            return None
        moved = changed(self.repo, base, main)
        if any(p.startswith(CHECK_TOOLING) for p in moved):
            return None
        for m in reversed(chain):
            if self.verdict(m.commit, base, drive).state == "running":
                if moved & changed(self.repo, base, m.commit):
                    return None
                return (f"`main` moved to {main[:12]} by files `{self.q.branch}` doesn't change while {m.commit[:12]}'s check runs: "
                        f"`{self.q.branch}` stays on {base[:12]} until it ends")
        return None

    def land(self, drive: dict, dry_run: bool) -> Member | None:
        """Land the newest member of `next` with a passing check (see the module's step 1): a fast-forward when `main` is still
        `next`'s base, so the members above it keep their commits and their checks; a merge commit onto a moved `main`."""
        q = self.q
        base, chain = self._chain()
        if not chain:
            return None
        main = _out(self.repo, "rev-parse", "origin/main")
        moved = sorted(changed(self.repo, base, main)) if main != base else []
        for i in range(len(chain) - 1, -1, -1):
            m = chain[i]
            v = self.verdict(m.commit, base, drive)
            if v.state != "passed":
                continue
            members = chain[: i + 1]
            prs = " ".join(f"#{x.pr}" for x in members)
            if moved and ((shared := sorted(set(moved) & changed(self.repo, base, m.commit)))
                          or (shared := [p for p in moved if p.startswith(CHECK_TOOLING)])):
                q.log(f"research queue drive: {prs} passed ({v.attempt}) on {base[:12]}, but `main` moved to {main[:12]} and "
                      f"{'shares' if set(shared) & changed(self.repo, base, m.commit) else 'changed check tooling in'} "
                      f"{', '.join(shared[:5])}: the next sync rebuilds `{q.branch}` on `main` for a new check")
                return None
            if dry_run or q.shadow:
                q.log(f"research queue drive: would land {prs} ({m.commit[:12]}, check {v.attempt})"
                      + (f" onto the moved `main` {main[:12]}" if moved else "") + (" [shadow]" if q.shadow else ""))
                return m
            _git(self.repo, "checkout", "-q", "-B", "main", "refs/remotes/origin/main")
            if moved:
                gained = _out(self.repo, "log", "--first-parent", "--format=%s", f"{base}..{main}").splitlines()
                body = (f"Merge {m.commit[:12]} ({' + '.join(x.head[:8] for x in members)})\n\n{prs}\n\n"
                        f"Checked: {v.attempt} (full check of {m.commit[:12]} = main {base[:12]} + "
                        f"{' + '.join(f'#{x.pr}@{x.head[:9]}' for x in members)}: passed)\n"
                        f"Skipped-check: main {base[:12]}..{main[:12]} ({'; '.join(gained)}) shares no files with {prs}\n")
                r = _git(self.repo, "merge", "--no-ff", "--no-edit", "-m", body, m.commit, check=False, env=IDENTITY)
            else:
                r = _git(self.repo, "merge", "-q", "--ff-only", m.commit, check=False)
            if r.returncode:
                _git(self.repo, "merge", "--abort", check=False)
                _git(self.repo, "reset", "-q", "--hard", "refs/remotes/origin/main")
                self.refuse(drive, f"land:{m.commit}", f"{prs} passed ({v.attempt}) but doesn't merge onto `main` {main[:12]}: "
                                                        f"{(r.stderr or r.stdout).strip()[-200:]}")
                return None
            p = q._remote(self.repo, "push", "-q", "--no-verify", "origin", "main:refs/heads/main", ok=None)
            if p.returncode:
                _git(self.repo, "reset", "-q", "--hard", "refs/remotes/origin/main")
                q.log(f"research queue drive: pushing `main` was refused ({p.stderr.strip()[-200:]}); the next tick tries again")
                return None
            tip = _out(self.repo, "rev-parse", "HEAD")
            q.log(f"research queue drive: landed {prs} on `main` at {tip[:12]} (check {v.attempt})")
            from .merge import _sweep_after, caught_up

            with authed(q.auth):
                caught_up(self.repo, "origin", "check", since=main, log=q.log,
                          runs={x.commit: v.attempt for x in members} | {tip: v.attempt})
                _sweep_after(self.repo, "main", "origin")
            return m
        return None

    # -- refusals

    def refuse(self, drive: dict, key: str, text: str, *handles: str | None) -> None:
        """Tell `CI` once per key, and ``handles``, the owning lanes of the PRs it concerns, beside it so each owner hears
        without a relay; with --shadow or --dry-run, only log it."""
        if key in drive["told"]:
            return
        to = tuple(dict.fromkeys((CI, *(f"@{h}" for h in handles if h))))
        if self.q.shadow or self.dry_run:
            self.q.log(f"research queue drive: would tell {' '.join(to)}: {text} [{'shadow' if self.q.shadow else 'dry run'}]")
            return
        if self.post(f"{' '.join(to)} queue: {text}", to):
            drive["told"].append(key)
        self.q.log(f"research queue drive: told {' '.join(to)}: {text}")

    def _handle(self, pr: int) -> str | None:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self.q.handle(self._prs[pr]) if pr in self._prs else None

    def culprit(self, members: list[Member], v: Verdict) -> tuple[Member | None, list[Member], str]:
        """The one member of a failed train whose own change (its merge commit against the one below) touches a file the
        failure points at (`implicated`), else None; with the members that touch one, and what the failure points at."""
        found: set[str] = set()
        for d in self.evidence(self.repo, v.attempt.split()):
            if (paths := implicated(d)) is None:
                return None, [], "a failure no test file accounts for"
            found |= paths
        if not found:
            return None, [], "no failed suite's report"
        hits = [(x, touches(changed(self.repo, f"{x.commit}^1", x.commit), found)) for x in members]
        hits = [(x, t) for x, t in hits if t]
        what = ", ".join(sorted({f for _, t in hits for f in t})) or ", ".join(sorted(found)[:4])
        return (hits[0][0] if len(hits) == 1 else None), [x for x, _ in hits], what

    def failed(self, drive: dict, m: Member, members: list[Member], v: Verdict) -> None:
        key, prs = f"failed:{m.commit}", " ".join(f"#{x.pr}" for x in members)
        if key in drive["told"]:
            return
        head = f"{m.commit[:12]} on `{self.q.branch}` ({prs}) failed its check: {v.why}"
        pick, hit, what = self.culprit(members, v)
        why = (f"{len(hit)} PRs ({' '.join(f'#{x.pr}' for x in hit)}) touch {what}" if hit else f"no PR's change is in {what}")
        if pick is not None:
            reason = f"train {m.commit[:12]} ({prs}), check {v.attempt}: the failure is in {what}, which only #{pick.pr} changes"
            if self.q.shadow or self.dry_run:
                self.q.log(f"research queue drive: would eject #{pick.pr}: {reason}")
            else:
                try:
                    self.q.eject(pick.pr, reason, "ci")
                except QueueError as e:
                    pick, why = None, f"only #{pick.pr} touches {what}, but ejecting it failed: {e}"
        if pick is not None:
            self.refuse(drive, key, f"{head}. Ejected #{pick.pr}: its change is the only one in {what}; the next sync rebuilds "
                                    f"without it. Fix it and re-mark it", self._handle(pick.pr))
            return
        self.refuse(drive, key, f"{head}. Not ejected: {why}. Eject the PR that broke it (`research queue eject N --reason ...`); "
                                "the next sync rebuilds without it", *(self._handle(x.pr) for x in hit))

    # -- the tick

    def tick(self, dry_run: bool = False) -> str:
        q = self.q
        self.dry_run, self._seen, self._prs = dry_run, {}, None
        q._clone()
        q.fetch()
        if q.store is not None and q.store.remote is not None:
            try:
                q.store.refresh(attempts_only=True)
            except Exception as e:  # noqa: BLE001  (the local catalog still answers)
                q.log(f"research queue drive: refreshing attempts failed ({type(e).__name__}: {e}); using the local catalog")
        state = q.load_state()
        drive = state.setdefault("drive", {})
        drive.setdefault("launched", {})
        drive.setdefault("told", [])
        landed = self.land(drive, dry_run)
        if landed is None and (why := self.held(drive)):
            if not dry_run:
                q.save_state(state)
            q.log(f"research queue drive: {why}")
            return why
        # before the sync: a rebuild takes a failed commit off `next`, and its failure would go untold (12 Oct 17:23Z: 3d11's)
        # and a train whose check is running, or has ended without its attempt in the store yet, keeps its place (19:47Z: a
        # rebuild in the seconds between 5897's end and its attempt's publication dropped a passing check)
        chain, main = q.chain(), _out(self.repo, "rev-parse", "origin/main")
        q.hold = 0
        for i, m in enumerate(chain):
            if m.commit not in drive["launched"]:
                continue
            v = self.verdict(m.commit, main, drive)
            if v.state == "failed":
                self.failed(drive, m, chain[: i + 1], v)
            elif v.state == "running":
                q.hold = i + 1
        members, entries = q.sync(dry_run)
        state = {**q.load_state(), "drive": drive}
        for n, c in state.get("conflicts", {}).items():
            self.refuse(drive, f"conflict:{n}@{c['head']}", f"#{n} at {c['head'][:8]} is out of `{q.branch}`: {c['why']}",
                        c.get("handle"))
        on_chain = {m.commit for m in members}
        for c, v in drive["launched"].items():
            if c not in on_chain and not dry_run:
                live = [rid for rid, (st, *_) in self._states(v["ids"]).items()
                        if st in ACTIVE and st != "unknown"]
                if live and v.get("failing"):
                    q.log(f"research queue drive: {' '.join(live)} of {c[:12]}, no longer on `{q.branch}`, keeps going for the "
                          f"full picture of its failure ({v['failing']})")
                elif live:
                    self.cancel(self.repo, live, f"drive: {c[:12]} is no longer on `{q.branch}` (rebuilt); no check of it lands")
                    q.log(f"research queue drive: cancelled {' '.join(live)}: {c[:12]} is no longer on `{q.branch}`")
        drive["launched"] = {c: v for c, v in drive["launched"].items() if c in on_chain}
        what = [f"landed #{landed.pr}"] if landed else []
        main = _out(self.repo, "rev-parse", "origin/main")
        running = sum(self.verdict(c, main, drive).state == "running" for c in drive["launched"])
        for i in self.trains(len(members)):
            m, prs = members[i], " ".join(f"#{x.pr}" for x in members[: i + 1])
            v = self.verdict(m.commit, main, drive)
            if v.state == "failed":
                self.failed(drive, m, members[: i + 1], v)
                what.append(f"{m.commit[:12]} failed")
                break                                       # every train above it holds the failure too
            if v.state == "passed":
                what.append(f"{m.commit[:12]} passed ({v.attempt}): it lands next tick")
                continue
            if v.state == "running":
                what.append(f"{m.commit[:12]} ({prs}): {v.attempt} running")
                continue
            if running >= self.max_running:
                what.append(f"{m.commit[:12]} waits: {running} checks of `{q.branch}` running")
                break
            agreement = needs_agreement(self.repo, main, m.commit)
            if dry_run or q.shadow:
                q.log(f"research queue drive: would launch train.sh {m.commit[:12]} {self.on}"
                      + (" --agreement" if agreement else "") + f" ({prs})" + (" [shadow]" if q.shadow else ""))
                what.append(f"would check {m.commit[:12]}")
                running += 1
                continue
            ids, tail = self.launch(self.repo, m.commit, self.on, agreement)
            if not ids:
                self.refuse(drive, f"launch:{m.commit}", f"train.sh {m.commit[:12]} {self.on} registered no run: {tail}")
                what.append("launch failed")
                break
            drive["launched"][m.commit] = {"ids": ids, "at": self.now(), "agreement": agreement}
            q.log(f"research queue drive: checking {m.commit[:12]} ({prs}): {' '.join(ids)}" + (" with agreement" if agreement else ""))
            what.append(f"launched {' '.join(ids)}")
            running += 1
        if not dry_run:
            state["ejected"] = {**state.get("ejected", {}), **q.load_state().get("ejected", {})}   # what `failed` ejected since
            q.save_state(state)
        q.log(f"research queue drive: {'; '.join(what) or 'nothing queued'}")
        return "; ".join(what) or "idle"

    def trains(self, n: int) -> list[int]:
        """The members whose commits get a check: every `train_size`-th one and the tip, each train stacked on the one below, so a
        failure holds up only the PRs from its train up, and the trains below it still land."""
        return sorted({*range(self.train_size - 1, n, self.train_size), n - 1}) if n else []
