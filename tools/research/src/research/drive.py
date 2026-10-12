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

What it refuses goes to the lander in the merge thread, once per commit or head: a failed check, a conflict, a check that ran
and passed but lacks a step the gate needs, a launch that registered no run.  The lander ejects the PR that broke it
(`research queue eject`), and the next sync rebuilds `next` without it.

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

from .queue import IDENTITY, Member, Queue, _git, _out

LANDER = "@old-circuits-and-proofs"
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


def run_states(repo: Path, ids: list[str]) -> dict[str, tuple[str, int | None, str | None]]:
    """Each run's (state, rc, ended_utc) from `research status RUN --json`; a run it can't read is ("unknown", None, None)."""
    out = {}
    for rid in ids:
        r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--json"], cwd=repo, capture_output=True, text=True)
        try:
            run = json.loads(r.stdout)["runs"][0]
            out[rid] = (run.get("state") or "unknown", run.get("rc"), run.get("ended_utc"))
        except (ValueError, KeyError, IndexError):
            out[rid] = ("unknown", None, None)
    return out


def cancel_runs(repo: Path, ids: list[str], reason: str) -> None:
    for rid in ids:
        subprocess.run([sys.executable, "-m", "research", "cancel", rid, "--by", "ci", "--reason", reason], cwd=repo,
                       capture_output=True, text=True)


def merge_thread_post(text: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "research", "msg", "send", "--as", "ci", "--thread", _merge_thread(), "--to", LANDER,
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


class Driver:
    def __init__(self, q: Queue, *, on: str, max_running: int = 2, train_size: int = 4,
                 launch: Callable[[Path, str, str, bool], tuple[list[str], str]] = train_launch,
                 states: Callable[[Path, list[str]], dict] = run_states, post: Callable[[str], bool] = merge_thread_post,
                 now: Callable[[], float] = time.time, cancel: Callable[[Path, list[str], str], None] = cancel_runs):
        self.q, self.on, self.max_running, self.train_size = q, on, max_running, train_size
        if launch is train_launch:
            launch = functools.partial(train_launch, auth=q.auth)
        self.launch, self.states, self.post, self.now, self.cancel = launch, states, post, now, cancel
        self.repo = q.workdir

    # -- what the store says of a commit

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
        states = self.states(self.repo, launch["ids"])
        if any(st in ACTIVE for st, _, _ in states.values()):
            return Verdict("running", " ".join(launch["ids"]))
        if all(rc == NOT_STARTED for _, rc, _ in states.values()):
            return Verdict("none", why="every twin exited without starting")
        ended = max((_utc(e) or launch["at"] for _, _, e in states.values()), default=launch["at"])
        if self.now() - ended < PUBLISH_GRACE_S:
            return Verdict("running", " ".join(launch["ids"]), "ended; its attempt isn't in the store yet")
        return Verdict("failed", " ".join(launch["ids"]),
                       f"{' '.join(launch['ids'])} ended ({', '.join(f'{s} rc={rc}' for s, rc, _ in states.values())}) "
                       f"and no attempt reached the store in {PUBLISH_GRACE_S // 60} min")

    # -- landing

    def _chain(self) -> tuple[str, list[Member]]:
        """(the base `next` was built on, its members oldest first), whether or not `main` has moved since."""
        ref = f"refs/remotes/origin/{self.q.branch}"
        if _git(self.repo, "rev-parse", "--verify", "-q", ref, check=False).returncode:
            return "", []
        base = _out(self.repo, "merge-base", "origin/main", ref)
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

    def refuse(self, drive: dict, key: str, text: str) -> None:
        """Tell the lander once per key; with --shadow, only log it."""
        if key in drive["told"]:
            return
        if self.q.shadow:
            self.q.log(f"research queue drive: would tell the lander: {text} [shadow]")
            return
        if self.post(f"{LANDER} queue: {text}"):
            drive["told"].append(key)
        self.q.log(f"research queue drive: told the lander: {text}")

    # -- the tick

    def tick(self, dry_run: bool = False) -> str:
        q = self.q
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
        members, entries = q.sync(dry_run)
        state = {**q.load_state(), "drive": drive}
        for n, c in state.get("conflicts", {}).items():
            self.refuse(drive, f"conflict:{n}@{c['head']}", f"#{n} at {c['head'][:8]} is out of `{q.branch}`: {c['why']}")
        on_chain = {m.commit for m in members}
        for c, v in drive["launched"].items():
            if c not in on_chain and not dry_run:
                live = [rid for rid, (st, _, _) in self.states(self.repo, v["ids"]).items()
                        if st in ACTIVE and st != "unknown"]
                if live:
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
                self.refuse(drive, f"failed:{m.commit}", f"{m.commit[:12]} on `{q.branch}` ({prs}) failed its check: {v.why}. "
                                                        "Eject the PR that broke it (`research queue eject N --reason ...`); "
                                                        "the next sync rebuilds without it")
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
            q.save_state(state)
        q.log(f"research queue drive: {'; '.join(what) or 'nothing queued'}")
        return "; ".join(what) or "idle"

    def trains(self, n: int) -> list[int]:
        """The members whose commits get a check: every `train_size`-th one and the tip, each train stacked on the one below, so a
        failure holds up only the PRs from its train up, and the trains below it still land."""
        return sorted({*range(self.train_size - 1, n, self.train_size), n - 1}) if n else []
