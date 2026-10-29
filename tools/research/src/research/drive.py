"""`research queue drive`: one tick of the landing automation (from an owner's ready mark to `main` moving, no agent acts
unless the automation refuses).  A loop runs it every few minutes.

What it refuses goes to `@ci` in the merge thread, once per commit or head, with the owner of the PR it concerns.  `--shadow`
drives `queue-shadow`: it lands nothing, launches nothing and posts nothing, and logs what it would do.

What only a PR's owner acts on goes to that owner alone (`tell`).  Each tick applies the stale rule (`research.stale`, drive.toml's
`stale_days` and `stale_close_days`: `_stale`): a PR unmarked at its head and idle `stale_days` rings its owner once a day, and
at `stale_close_days` drive closes it, its branch kept.  After a landing, the owner of each unmarked open PR whose head contains
a landed PR's head hears once that its base landed and to merge `main` (`_stacked`).

Review landing, the only landing.  `main` holds only fully checked commits: PRs land on a review onto the
landing branch, drive.toml's `land_into` (`checked`: `Queue.into`), drive runs the full check on its tip, and `main` is
fast-forwarded only to a commit of it whose after-landing run passed every step on the merge gate (`_promote`: not
circuit-check, ci.toml's `gate = false`, whose red reverts its culprit on `checked`); with `land_into = "main"` they
land on `main` and are checked after.  A tick
syncs `next` (`Queue.sync`: admit and drop PRs, rebuild on a moved `checked`), then lands its longest prefix whose every member has a land verdict at its head and no hold
(`research queue review N --land|--hold`, a `reviewed` label on `pr:N@HEAD`; one by its owner, the owner's handle or its ready
mark's author doesn't count: `queue.unreviewed`), a fast-forward of `checked` (`_land`); once `main` reaches it, `merge.caught_up`
(the spec DM, node 1's mirror, the post-train regression, the Lean cache publish, the landing post) and the sweep (`_on_main`).
A `main` pushed outside drive (a pure move, on its own check) is merged into `checked` (`_absorb`).  No member waits for a
Lean check of its own: a PR enters `next` only once its preflight passed on its merge there (`queue.Queue.preflight`: the
merged tree's checks and the build of the Lean modules it touches, onto the member below its place), so one PR's failure
keeps out only that PR, and the after-landing run's lean-audit checks Lean.  The sync moves the members with a land verdict
below those waiting for one (drive.toml's `reviewed_first`: `_lift`; never past a held member), where they land that tick, so
a reviewed member seldom waits above an unreviewed one.  And every .py file the landing leaves must compile (`py_compile`):
the PR that last changed one that doesn't is ejected.  Each landing awaits the after-landing run, one at a
time of `checked`'s tip (`train.sh TIP MACHINES --deferred STEPS --base VERIFIED`) with every ci.toml job (`AFTER_LANDING`),
circuit-check and ratchet-gate among them, which `_promote` doesn't wait for (`gate = false`: `_gating`), its base the last
verified `checked` commit, below the tip, for ratchet-gate to judge the
landings above; with drive.toml's `after_landing_shards` N, as N runs at once (`--shard K/N` each), passed once every shard
passed; `--no-cache --circuit-cache` while `after_landing_no_cache` (no pass reused but circuit-check's per-target ones), and
`--no-regenerate` (a lock record difference fails it).  Nothing lands before one such run of `checked`
itself ended (its baseline, verifying the steps it passed; `/data/infra/drive.sh queue baseline SHA --by ci` records a green one
by hand).  A red run runs on, so every red suite is named, and is cancelled only once its `TAIL` is all it has left (a cancel
earlier in a ~4 h run leaves the rest unknown).  Each failed suite or step (`_red`, `postland["red"]`) gets a culprit
among the landings since it last passed (`postland["green"]`, per suite or step, so a `checked` never verified whole still names
one): for ratchet-gate, the merge the gate's log says it fails; else the landing whose own diff touches what the failure
names (`suite_paths`: failing tests, its log's failures, the read guard's `outside`), else its declared inputs
(`_input_rule`); several are bisected by landing, each step a run of one landing's commit judged by that suite alone, while
landing goes on.  The ring (`_ring`) quotes the run, the first error line
(`first_error`) and what a cancel stopped (`not_run`, "not run"), to @ci, the suite's owner and the culprit's; the culprit is
reverted on `checked`, never on `main`, `FIX_WINDOW_S` after it (`_red_clock`) unless a fix is marked (`_fix`); a red
circuit-check's culprit at once (`CIRCUIT_RED`), and while circuit-check is red on `checked` no PR whose own change touches
its inputs lands (`_circuit_held`), every other one landing on.  A red run moves nothing on `main`.  While the oldest landing no run verified is
`BEHIND_S` old, @ci and @top are rung every `BEHIND_EVERY_S` (`_behind`).  A run with no readable state `LOST_AFTER_S` after its
launch is lost.

The post-landing runs (the after-landing run and its bisection's) are launched detached (`detach`), so no tick waits on
train.sh, which takes minutes to print a run's ids.  A later tick reads the ids from its log (`launched`, `_resolve`); until
then the launch counts as running, and a train.sh that exits without them registered no run.

State (launches, refusals told) is the queue's `state.json`, under `drive`."""
from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .queue import (IDENTITY, PR, Member, Queue, QueueError, StoreLabels,
                    _git, _out, built_on, changed, forces_cold, unreviewed)

#: who hears what drive can't settle
CI = "@ci"
#: a pytest location (`tests/test_x.py:503: NameError`) or a traceback frame (`File "/.../x.py", line 12`) in a suite's log
TRACE_RE = re.compile(r'^\s*([\w./+-]+\.py):\d+:|File "([^"]+\.py)", line \d+', re.M)
#: a repository path a pytest failure report names (`E   tools/x/README.md: 48 KB > 48 KB`: a lint's offender is in
#: its assertion, not its traceback); relative, with a directory
NAMED_RE = re.compile(r"(?<![\w./-])((?:[\w+-]+/)+[\w+.-]*\w\.\w+)")
#: pytest's failing tests in its short summary
FAILED_RE = re.compile(r"^FAILED ([^\s:]+)::", re.M)
#: a test module pytest couldn't collect, in its short summary (`ERROR tests/x.py - KeyError: ...`)
ERROR_RE = re.compile(r"^ERROR ([^\s:]+\.py)", re.M)
#: in lean-audit's log: the package an audit failed (its path under the audit's scratch tree), and a Lean error's file, relative
#: to that package
LEAN_AUDIT_RE = re.compile(r"^AUDIT \S*?/lean-audit-scratch-[^/\s]+/(\S+?): FAIL", re.M)
LEAN_ERROR_RE = re.compile(r"^\s*error: (\S+?\.lean):\d+", re.M)
#: the audit's own failures under its `AUDIT .../PKG: FAIL` line: a file no root imports (relative to PKG), and a module that
#: imports outside its layer (neither names a file drive's other patterns match)
LEAN_ORPHAN_RE = re.compile(r"^\s+orphan: (\S+?\.lean) is not imported")
LEAN_LAYERS_RE = re.compile(r"^\s+layers: ([A-Za-z][\w.]*) imports ")
#: a run's or a step's exit that a kill gave, not a failure of what it checked: SIGHUP, SIGKILL, SIGTERM as 128+N, or a signal
KILLED_RC = (129, 137, 143)
#: what the Lean check says of a process of its own that was killed (`tools/verity/lean/check.py`: the replay, `--fresh`)
KILLED_SAYS = "not a kernel verdict"
#: circuit-check's report of a failure: `FAIL <check> <target>`, and `circuit-check/killed` for a target whose process died
CIRCUIT_FAIL_RE = re.compile(r"^FAIL (\S+)", re.M)
CIRCUIT_KILLED = "circuit-check/killed"
#: how often drive sweeps the repository's branches (`merge.sweep_branches`)
BRANCH_SWEEP_S = 86400
#: how often drive applies the stale rule (`Driver._stale`), which reads the labels of every idle PR: its rings are daily
STALE_EVERY_S = 3600
#: Lean's own inputs, for a Lean step that failed with no file named
LEAN_INPUTS = (".lean", "lakefile.lean", "lakefile.toml", "lean-toolchain", "lake-manifest.json", "lean-audit.json")
#: states `research status` reports for a run that may still publish an attempt
ACTIVE = ("submitted", "waiting", "running", "unknown")
#: check.py's exit for a twin that never took a slot
NOT_STARTED = 75
#: the steps of a check (`tools/verity/check/check.py`)
CHECK_STEPS = ("preflight-lock", "preflight-lints", "preflight-fixtures", "lean-changed", "flock-circuit-build", "lean-build",
               "lean-unit-cut", "lean-agreement", "lean-audit", "lean-suites", "pytest", "circuit-check", "flock-rust-tests",
               "ratchet-gate")
#: the check step that runs the vLLM integration's ratchet gate on the merges above the check's base
RATCHET_STEP = "ratchet-gate"
#: the ratchet gate's line for a change it fails (`ratchet_gate.render`): its name (`#N`, or a range of commits), its top
#: commit's first 12 digits, and why
RATCHET_FAILED_RE = re.compile(r"^  (\S+) \(([0-9a-f]{12})\): failed: (.*)$", re.M)
#: what a review landing's verdict says
REVIEWED = "a land verdict at each head"
#: the keys of drive's state that only train landing and the per-member Lean checks kept, dropped from a state that has them
TRAIN_STATE = ("launched", "deferred", "resent", "base", "bisect", "orphans", "per_pr", "prewarm", "lean", "marks")
#: the after-landing run of a review landing: every ci.toml job (`--deferred` takes jobs; the preflights are none), those off the
#: merge gate (`gate = false`: circuit-check, ratchet-gate) among them, which no check before landing runs
AFTER_LANDING = tuple(s for s in CHECK_STEPS if not s.startswith("preflight-"))
#: the after-landing steps a run may skip by name: lean-agreement when nothing it checks changed, lean-changed under --no-cache
AFTER_LANDING_OPTIONAL = ("lean-agreement", "lean-changed")
#: how long a run may show no readable state (`unknown`) before drive takes it as lost (`Driver._lost`)
LOST_AFTER_S = 3 * 3600
#: how long the owners rung for a red suite or step have to mark a fix before drive reverts its culprit (`Driver._red_clock`;
#: without it PRs keep landing on a `main` no after-landing run passed, and its failures reach only @ci)
FIX_WINDOW_S = 3600
#: the red record reverted at once, with no `FIX_WINDOW_S`, and while it stands no PR touching what it reads lands
#: (`Driver._circuit_held`): circuit-check runs only after landing, so every landing on its red would go unchecked
CIRCUIT_RED = "step circuit-check"
#: how far behind `main`'s tip its last verified commit may fall before drive rings @ci and @top, and how often it rings again
#: while it stays there (`Driver._behind`)
BEHIND_S, BEHIND_EVERY_S = 3 * 3600, 3600
#: how long an after-landing run may run before its running no longer explains that lag (a cold one takes about 4 h,
#: circuit-check most of it)
POSTLAND_RUN_S = 5 * 3600
#: the steps a red after-landing run is cancelled in, once only they are left: they name no suite and check no Lean, and
#: circuit-check alone is most of a cold run's ~4 h (the measurement is in `follow_postland`'s docstring)
TAIL = ("circuit-check", "flock-rust-tests")
#: who the alarm on a `main` long unverified rings beside `CI` (`Driver._behind`)
TOP = "@top"
#: a run stopped by SIGTERM (128 + 15): a cancel, or a twin whose sibling won the slot
SIGTERM_RC = 143
RUN_RE = re.compile(r"^RUN ((?:r20\d{6}-\d{6}-[0-9a-f]{4} ?)+)$", re.M)
#: what `git revert` writes of the commit it undoes
REVERTS_RE = re.compile(r"^This reverts commit ([0-9a-f]{40})", re.M)


@dataclass
class Verdict:
    state: str                    # passed | running | failed | none
    attempt: str = ""
    why: str = ""


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
    (`merge.caught_up`, `_sweep_after`), whose git children read os.environ (a git that prompts, as `caught_up`'s `ls-remote` in the
    queue's clone would, waits on a password at the tty and holds drive)."""
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


def train_launch(repo: Path, commit: str, machine: str, agreement: bool, auth: tuple[str, ...] = (),
                 deferred: str | None = None, no_cache: bool = False, no_regenerate: bool = False,
                 detached: bool = False, circuit_cache: bool = False, shard: str | None = None,
                 base: str | None = None) -> tuple[list[str], str] | dict:
    """`train.sh COMMIT MACHINE [--agreement] [--base BASE] [--shard K/N]` from the queue's clone: (the run ids it printed,
    its output's tail); ``detached``, started in the background (`detach`) from a copy of train.sh, which reads nothing by its
    own path (``$0`` only for its usage) and finds the tree from its working directory, and returned at once as the record
    `launched` reads.  Git never prompts (the queue's clone has no credential helper, so train.sh's fetch would wait on a
    password at the tty)."""
    argv = ["bash", "tools/verity/check/train.sh", commit, machine, *(["--agreement"] if agreement else []),
            *(["--deferred", deferred] if deferred else []), *(["--no-cache"] if no_cache else []),
            *(["--circuit-cache"] if circuit_cache else []), *(["--no-regenerate"] if no_regenerate else []),
            *(["--base", base] if base else []), *(["--shard", shard] if shard else [])]
    env = {**os.environ, **auth_env(auth), "GIT_TERMINAL_PROMPT": "0", "TRAIN_AS": "ci"}
    if detached:
        return detach(repo, argv, env, script=argv[1])
    r = subprocess.run(argv, cwd=repo, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    m = RUN_RE.search(r.stdout)
    return (m[1].split() if m else []), (r.stdout + r.stderr).strip()[-600:]


#: where detached launches (`detach`) keep their output and exit, under drive's TMPDIR
LAUNCHES = "drive-launches"
#: how long a detached launch may print no run ids before drive takes it as having registered none: train.sh stops its launcher
#: 30 min into its wait for one (`launch_wait_s`), and a restart of the pod can leave its pid to another process
LAUNCH_LOST_S = 3600
#: a detached launch's job: the command's output to `log` in its directory, then its exit status to `rc`
DETACH_JOB = 'd=$1; shift; "$@" > "$d/log" 2>&1; echo "$?" > "$d/rc.tmp" && mv "$d/rc.tmp" "$d/rc"'


def detach(repo: Path, argv: list[str], env: dict[str, str], script: str | None = None) -> dict:
    """``argv`` in ``repo``, in a session of its own, so no tick waits on it (train.sh takes minutes to print a Lean
    check's run ids): its output and exit go to a directory of their own under
    `LAUNCHES`.  ``script``, a path of ``repo`` in ``argv``, runs from a copy there: bash reads a script as it runs, and a later
    tick resets and checks out ``repo``.  Returns the record `launched` reads."""
    root = Path(tempfile.gettempdir()) / LAUNCHES
    root.mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(prefix="train-", dir=root)
    if script:
        copy = shutil.copy(repo / script, Path(d) / Path(script).name)
        argv = [str(copy) if a == script else a for a in argv]
    p = subprocess.Popen(["sh", "-c", DETACH_JOB, "sh", d, *argv], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"dir": d, "pid": p.pid}


def _ids(rec: dict) -> str:
    """A launch record's run ids, for a message, while its train.sh runs detached too."""
    return " ".join(rec.get("ids") or ()) or ("(launching)" if rec.get("launch") or rec.get("shards") else "(no run)")


def launched(launch: dict) -> tuple[list[str] | None, str]:
    """(the run ids a detached train.sh (`detach`) printed, [] once it exited without them or its process is gone without an
    exit, None while it runs; its output's tail)."""
    d = Path(launch["dir"])
    try:
        out = (d / "log").read_text(errors="replace")
    except OSError:
        out = ""
    tail = out.strip()[-600:]
    if m := RUN_RE.search(out[: out.rfind("\n") + 1]):          # a whole line: train.sh may be writing it
        return m[1].split(), tail
    if (d / "rc").is_file():
        return [], tail
    try:
        os.kill(int(launch["pid"]), 0)
    except ProcessLookupError:
        return [], f"{tail} (its process is gone with no exit written: a restart of the pod)".strip()
    except (PermissionError, ValueError, TypeError, KeyError):
        pass
    return None, tail


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


def run_states(repo: Path, ids: list[str]) -> dict[str, tuple]:
    """Each run's (state, rc, ended_utc, what has failed so far while it runs: `failing`, the machine it runs on) from `research
    status RUN --json`, which fetches `PROGRESS` with its records (a check waiting in its machine's slot line reads `waiting`); a run
    it can't read is ("unknown", None, None, None)."""
    also = [x for n in PROGRESS for x in ("--also", n)]

    def one(rid: str) -> tuple:
        r = subprocess.run([sys.executable, "-m", "research", "status", rid, "--json", *also], cwd=repo, capture_output=True, text=True)
        try:
            run = json.loads(r.stdout)["runs"][0]
            state = run.get("state") or "unknown"
            return (state, run.get("rc"), run.get("ended_utc"),
                    failing(Path(run["dir"])) if state in ACTIVE and run.get("dir") else None, (run.get("remote") or {}).get("machine"))
        except (ValueError, KeyError, IndexError):
            return ("unknown", None, None, None)

    return dict(zip(ids, _each(one, ids)))


#: how many runs drive reads at once: each `research status` is an ssh round trip to the run's node (reading
#: a few dozen one at a time takes a tick half an hour)
STATUS_JOBS = 8


def _each(fn, items: list, jobs: int = STATUS_JOBS) -> list:
    """``fn`` of each of ``items``, ``jobs`` at a time, in their order."""
    if len(items) < 2:
        return [fn(x) for x in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, items))


def cancel_runs(repo: Path, ids: list[str], reason: str) -> dict[str, str]:
    """`research cancel` of each of ``ids``: {run id: why} of each cancel that failed (a cancel that fails unseen leaves its run holding
    a machine's line for hours)."""
    failed = {}
    for rid in ids:
        r = subprocess.run([sys.executable, "-m", "research", "cancel", rid, "--by", "ci", "--reason", reason], cwd=repo,
                           capture_output=True, text=True)
        if r.returncode:
            failed[rid] = (r.stderr or r.stdout).strip()[-300:] or f"exit {r.returncode}"
    return failed


def merge_thread_post(text: str, to: tuple[str, ...] = (CI,), sender: str = "ci") -> bool:
    """Post ``text`` to the merge thread as ``sender``, addressed to ``to`` but not the sender, and with the sender dropped from
    the names ``text`` starts with: `research msg send` refuses a `--to` naming the sender and a post that starts with its
    sender's name alone."""
    others = [t for t in to if t != f"@{sender}"]
    if lead := re.match(r"(?:@[\w.-]+ )+", text):
        text = " ".join([*(n for n in lead[0].split() if n != f"@{sender}"), text[lead.end():]])
    r = subprocess.run([sys.executable, "-m", "research", "msg", "send", "--as", sender, "--thread", _merge_thread(),
                        *(["--to", *others] if others else []), "--quiet", "--report", "--text", text], capture_output=True, text=True)
    return r.returncode == 0


def _merge_thread() -> str:
    from .monitors import MERGE_THREAD

    return MERGE_THREAD


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
    from .merge import fetch_run

    got = fetch_run(rid, names, repo)
    return got[0] if got else None


def evidence(repo: Path, ids: list[str]) -> list[Path]:
    """The local run dirs of ``ids`` with their steps, suite reports and failed suites' logs fetched (`research status --also`)."""
    def one(rid: str) -> Path | None:
        if (d := _fetch(repo, rid, [*PROGRESS, "suites.json", "suites-lean.json"])) is None:
            return None
        logs = [f"suites/{Path(s['log']).name}" for s in failed_suites(d) if s.get("log")]
        logs += [Path(st["log"]).name for st in failed_steps(d) if st.get("log") and st.get("step") not in ("pytest", "lean-suites")]
        if logs:
            _fetch(repo, rid, logs)
        return d

    return [d for d in _each(one, ids) if d is not None]


def failed_steps(run_dir: Path) -> list[dict]:
    out = []
    with contextlib.suppress(OSError):
        for line in (run_dir / PROGRESS[0]).read_text().splitlines():
            with contextlib.suppress(ValueError):
                if (rec := json.loads(line)).get("status") == "failed":
                    out.append(rec)
    return out


def suite_inputs(repo: Path, commit: str, suite: str) -> list[str]:
    """The paths suite ``suite`` (its directory) reads at ``commit``: its directory and its `[tool.verity.tests] inputs`; the
    repository's own suite (``.``) only its inputs."""
    import tomllib

    own = "pyproject.toml" if suite.strip("/") in ("", ".") else f"{suite.strip('/')}/pyproject.toml"
    r = _git(repo, "show", f"{commit}:{own}", check=False)
    try:
        inputs = tomllib.loads(r.stdout).get("tool", {}).get("verity", {}).get("tests", {}).get("inputs", []) if not r.returncode else []
    except tomllib.TOMLDecodeError:
        inputs = []
    return [i for i in ([] if suite in ("", ".") else [suite]) + [str(x) for x in inputs] if i]


def under(f: str, paths: list[str]) -> bool:
    return any(f == p or f.startswith(p.rstrip("/") + "/") for p in paths)


def lean_errors(log: str) -> set[str]:
    """The repository paths of the Lean files lean-audit's ``log`` reports errors in: each `error: FILE.lean:LINE`, each
    `orphan: FILE.lean` and the module of each `layers: MODULE imports ...` after an `AUDIT .../PKG: FAIL` line, FILE relative to
    PKG."""
    out, pkg, failures = set(), None, False
    for line in log.splitlines():
        if m := LEAN_AUDIT_RE.match(line):
            pkg, failures = m[1], False
        elif pkg and ((m := LEAN_ERROR_RE.match(line)) or (m := LEAN_ORPHAN_RE.match(line))):
            out.add(os.path.normpath(os.path.join(pkg, m[1])))
        elif pkg and (m := LEAN_LAYERS_RE.match(line)):
            parts = m[1].split(".")
            root = os.path.dirname(pkg) if parts[0] == os.path.basename(pkg) else pkg
            out.add(os.path.normpath(os.path.join(root, *parts) + ".lean"))
        elif pkg and "Some required targets logged failures:" in line:
            failures = True                                 # lake's own list: the step's log keeps it when the errors scroll by
        elif pkg and failures and (m := re.match(r"^\s*- ([A-Za-z][\w.]*)\s*$", line)):
            parts = m[1].split(".")                         # a module of a package rooted at its own name sits beside it
            root = os.path.dirname(pkg) if parts[0] == os.path.basename(pkg) else pkg
            out.add(os.path.normpath(os.path.join(root, *parts) + ".lean"))
        elif failures and not line.strip().startswith("-"):
            failures = False
    return out


def killed(run_dir: Path, why: str = "") -> bool:
    """Whether a check failed by being killed rather than by what it checked: a failed step that exited by a signal or 128+N,
    or the run's own exit (``why``, the verdict's `rc=N`) one of those."""
    steps = failed_steps(run_dir)
    if any(isinstance(rc, int) and (rc < 0 or rc in KILLED_RC) for rc in (st.get("rc") for st in steps)):
        return True
    for st in steps:                                     # a step that says one of its own processes was killed (the
        if st.get("log"):                                # Lean check's kernel replay, exit 137)
            with contextlib.suppress(OSError):
                if KILLED_SAYS in (run_dir / Path(st["log"]).name).read_text(errors="replace"):
                    return True
    # circuit-check alone failed, and only on targets whose process died (out of memory, say): the machine's fault, not a
    # landing's
    if len(steps) == 1 and steps[0].get("step") == "circuit-check" and steps[0].get("log"):
        with contextlib.suppress(OSError):
            fails = CIRCUIT_FAIL_RE.findall((run_dir / Path(steps[0]["log"]).name).read_text(errors="replace"))
            if fails and all(f == CIRCUIT_KILLED for f in fails):
                return True
    return any(f"rc={rc}" in why for rc in (*KILLED_RC, -9, -15))


def _named(log: str, base: str) -> set[str]:
    """The files a pytest log's failures name: its tracebacks' frames, and the repository paths in its failure report."""
    rel = lambda p: p if p.startswith("/") else os.path.normpath(os.path.join(base, p))  # noqa: E731
    paths = {rel(m[1] or m[2]) for m in TRACE_RE.finditer(log)}
    report = log.partition("= FAILURES =")[2].partition("short test summary info")[0]
    return paths | {rel(m[1]) for m in NAMED_RE.finditer(report)}


def _step_log(run_dir: Path, st: dict) -> str:
    try:
        return (run_dir / Path(st.get("log") or "").name).read_text(errors="replace") if st.get("log") else ""
    except OSError:
        return ""


def _suite_log(run_dir: Path, s: dict) -> str:
    if s.get("log"):
        with contextlib.suppress(OSError):
            return (run_dir / "suites" / Path(s["log"]).name).read_text(errors="replace")
    return ""


def step_paths(run_dir: Path, st: dict) -> set[str] | None:
    """The paths one failed step other than pytest and lean-suites points at (`implicated`), None when its log names none."""
    log = _step_log(run_dir, st)
    if st.get("step") == "lean-audit" and (lean := lean_errors(log)):
        return lean
    if not (tests := FAILED_RE.findall(log)):
        return None
    return {os.path.normpath(t) for t in tests} | _named(log, "")


def ratchet_failures(run_dir: Path, st: dict) -> list[tuple[str, str, str]]:
    """(name, top commit's first 12 digits, why) of each change a ratchet-gate step's log says the gate failed."""
    return RATCHET_FAILED_RE.findall(_step_log(run_dir, st))


def suite_paths(run_dir: Path, s: dict) -> set[str] | None:
    """The paths one failed suite's record points at (`implicated`), None when it names no failing test and no outside read."""
    base, tests = s.get("suite") or "", s.get("failed_tests") or []
    log = _suite_log(run_dir, s)
    if not tests and (errors := ERROR_RE.findall(log)):     # a collection error names no failing test, only its module,
        tests = errors                                       # and blaming the suite's inputs would take every landing
    if not tests:
        out, child = s.get("outside") or {}, s.get("child_outside") or {}
        modules = list(out.values() if isinstance(out, dict) else ()) + list(child)
        files = list(out) + [f for fs in child.values() for f in fs]
        if not files:
            return None
        return {os.path.normpath(p) for p in files} | {os.path.normpath(m) for m in modules if m and m != "*"}
    paths = {os.path.normpath(os.path.join(base, t.split("::")[0])) for t in tests}
    return paths | (_named(log, base) if log else set())


def implicated(run_dir: Path) -> set[str] | None:
    """The paths a failed check's failure points at: each failed suite's failing test files and the files its log's failures
    name (`_named`: relative to the repository, or absolute as the log wrote them), and the same of a failed step that runs
    pytest on the repository's own tests (`preflight-lints`). A suite the read guard failed with no failing test names the
    files its tests (or their children, `child_outside`) read outside its inputs and the test modules that read them (its
    record's `outside`, path -> module), so the PR that added the reading test, or changed the file read, is implicated.
    None when another step failed, or a failed suite or step names no failing test and no outside read (a timeout, a kill),
    so no file says what broke it."""
    paths: set[str] = set()
    for st in failed_steps(run_dir):
        if st.get("step") in ("pytest", "lean-suites"):
            continue
        if (got := step_paths(run_dir, st)) is None:
            return None
        paths |= got
    suites = failed_suites(run_dir)
    if not suites and not paths:
        return None
    for s in suites:
        if (got := suite_paths(run_dir, s)) is None:
            return None
        paths |= got
    return paths


#: the step whose suites each report of a check holds: what a bisection of one suite runs
SUITE_REPORTS = {"suites.failed.json": "pytest", "suites-lean.failed.json": "lean-suites", "suites.json": "pytest",
                 "suites-lean.json": "lean-suites"}
#: pytest's failure line, and a tool's error line, in a log: what a ring quotes of it
ERROR_LINE_RE = re.compile(r"^(?:E\s{2,}\S.*|\s*(?:error|Error|ERROR)\b.*|FAIL\b.*)$", re.M)


def suite_rows(run_dir: Path) -> dict[str, dict]:
    """{name: record} of each suite a check's reports name, with the step that ran it (`SUITE_REPORTS`): its failures so far,
    and once the check ended, every suite with its status."""
    out: dict[str, dict] = {}
    for name, step in SUITE_REPORTS.items():
        for s in _docs(run_dir, name):
            if n := s.get("name") or s.get("suite"):
                out[n] = {**out.get(n, {}), **s, "step": step, "status": s.get("status") or "failed"}
    return out


def step_records(run_dir: Path) -> list[dict]:
    out = []
    with contextlib.suppress(OSError):
        for line in (run_dir / PROGRESS[0]).read_text().splitlines():
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
    return out


def not_run(st: dict) -> bool:
    """A step that found nothing: stopped (check.py's `stopped`, a cancel's), or killed by a SIGTERM before check.py marked it
    so (it reads `failed` with an empty log)."""
    return st.get("status") == "stopped" or (st.get("status") == "failed" and st.get("rc") in (-15, SIGTERM_RC))


def first_error(run_dir: Path, rec: dict) -> str:
    """What a ring quotes of a failed suite or step: its failed test ids, or the paths the read guard caught it reading outside
    its inputs, and its log's first error line."""
    tests = rec.get("failed_tests") or []
    reads = sorted({*(rec.get("outside") or {}), *(f for fs in (rec.get("child_outside") or {}).values() for f in fs)})
    what = (f"failed {', '.join(tests[:3])}" + (f" and {len(tests) - 3} more" if len(tests) > 3 else "") if tests else
            f"read outside its declared inputs: {', '.join(reads[:3])}" + (f" and {len(reads) - 3} more" if len(reads) > 3 else "")
            if reads else "")
    log = _suite_log(run_dir, rec) if "suite" in rec or "name" in rec else _step_log(run_dir, rec)
    line = m[0].strip()[:240] if (m := ERROR_LINE_RE.search(log)) else ""
    return "; ".join(x for x in (what, f"`{line}`" if line else "") if x) or rec.get("detail") or "its log has no error line"


def touches(files: set[str], paths: set[str]) -> set[str]:
    """The ``files`` (repository paths) among ``paths``, an absolute path matching the file it ends in when that file is under a
    directory (a bare `conftest.py` would match every conftest an installed package has)."""
    return {f for f in files for p in paths if p == f or ("/" in f and p.startswith("/") and p.endswith("/" + f))}


class Driver:
    def __init__(self, q: Queue, *, on: str,
                 launch: Callable[[Path, str, str, bool], tuple[list[str], str]] = train_launch,
                 states: Callable[[Path, list[str]], dict] = run_states, post: Callable[[str, tuple[str, ...]], bool] = merge_thread_post,
                 now: Callable[[], float] = time.time, cancel: Callable[[Path, list[str], str], dict[str, str] | None] = cancel_runs,
                 evidence: Callable[[Path, list[str]], list[Path]] = evidence,
                 after_landing_no_cache: bool = False, reviews: Callable[[int, str], list] | None = None,
                 ready_by: Callable[[int, str], str | None] | None = None,
                 stale_days: int = 0, stale_close_days: int = 0, stale_keep: tuple[int, ...] = (),
                 after_landing_shards: int = 1):
        self.q, self.on = q, on
        #: the after-landing run as this many runs at once, shard K of ci.toml's plan with N shards each (`start_postland`)
        self.after_landing_shards = after_landing_shards
        #: drive.toml's stale rule (`research.stale`, `_stale`): ring a stale PR's owner once a day, close it at the second
        self.stale_days, self.stale_close_days, self.stale_keep = stale_days, stale_close_days, stale_keep
        #: each member lands on `checked` on a reviewer's verdict, and the full check of
        #: `checked` decides what reaches `main` (`_review`, `_promote`), reusing no pass with ``after_landing_no_cache`` but
        #: circuit-check's per-target ones (`--circuit-cache`)
        self.after_landing_no_cache = after_landing_no_cache
        #: the `reviewed` labels on a PR's head (`StoreLabels.reviews`)
        self.reviews: Callable[[int, str], list] = reviews or (lambda pr, head: StoreLabels(q.store, q.log).reviews(pr, head))
        #: who wrote the ready mark on a PR's head (`StoreLabels.ready_by`), one of those whose verdict doesn't count (`_owners`)
        self.ready_by: Callable[[int, str], str | None] = ready_by or (lambda pr, head: StoreLabels(q.store, q.log).ready_by(pr, head))
        #: the after-landing run on the landing branch's tip (`postland`): `train.sh COMMIT MACHINE --deferred STEPS`, detached
        #: (`_started`); an injected `launch` launches it too
        self.launch_postland: Callable[..., tuple[list[str], str] | dict] = (
            (lambda repo, commit, machine, steps, agreement=False, no_cache=False, circuit_cache=False, no_regenerate=False,
                    shard=None, base=None:
             train_launch(repo, commit, machine, agreement, auth=q.auth, deferred=steps, no_cache=no_cache,
                          no_regenerate=no_regenerate, detached=True, circuit_cache=circuit_cache, shard=shard, base=base))
            if launch is train_launch else (lambda repo, commit, machine, steps, **_: self.launch(repo, commit, machine, False)))
        #: what a detached launch has printed so far (`launched`, `_resolve`)
        self.launched: Callable[[dict], tuple[list[str] | None, str]] = launched
        if launch is train_launch:
            launch = functools.partial(train_launch, auth=q.auth)
        self.launch, self.states, self.post, self.now, self.cancel, self.evidence = launch, states, post, now, cancel, evidence
        self.repo, self.dry_run, self._seen, self._prs, self._ev = q.workdir, False, {}, None, {}
        self._drive: dict | None = None

    # -- what the store says of a commit

    def _states(self, ids: list[str]) -> dict:
        """`states` of ``ids``, each run read once a tick: one `research status` is a few seconds over ssh, and a tick
        reads many."""
        if missing := [rid for rid in ids if rid not in self._seen]:
            self._seen.update(self.states(self.repo, missing))
        return {rid: self._seen[rid] for rid in ids}

    @staticmethod
    def _started(got: tuple[list[str], str] | dict) -> tuple[dict | None, str]:
        """A launch record's fields from what `launch_postland` returned: {"ids"} of a launch that printed
        them, {"ids": [], "launch"} of one still running detached (`detach`), which `_resolve` reads on later ticks and which
        counts as running until then; None, with train.sh's output's tail, of one that registered no run."""
        if isinstance(got, dict):
            return {"ids": [], "launch": got}, ""
        ids, tail = got
        return ({"ids": ids} if ids else None), tail

    def _resolve(self, rec: dict, what: str) -> str | None:
        """Read a detached launch's run ids into ``rec`` (`launched`) once train.sh has printed them, its ``at`` then.  None
        while it runs (``rec["launch"]`` stays) or once ``rec`` has its ids; else, its ``launch`` dropped, the tail of a train.sh
        that registered no run, or printed none within `LAUNCH_LOST_S`."""
        if not (launch := rec.get("launch")) or self.dry_run or self.q.shadow:
            return None
        ids, tail = self.launched(launch)
        if ids is None and self.now() - rec.get("at", self.now()) <= LAUNCH_LOST_S:
            return None
        del rec["launch"]
        if not ids:
            return (tail if ids is not None else f"no run ids {LAUNCH_LOST_S // 60} min after its launch: {tail}") + \
                f" (log {launch['dir']}/log)"
        rec.update(ids=ids, at=self.now())
        shutil.rmtree(launch["dir"], ignore_errors=True)
        self.q.log(f"research queue drive: {what}: {' '.join(ids)}")
        return None

    def _evidence(self, v: Verdict) -> list[Path]:
        """`evidence` of a failed verdict, fetched once a tick: `_red` and `_tail_left` read the same run's."""
        if v.attempt not in self._ev:
            self._ev[v.attempt] = self.evidence(self.repo, v.attempt.split())
        return self._ev[v.attempt]

    def _undone(self, c: str) -> bool:
        """Whether a revert on the landing branch (`Queue.into`) above ``c`` undid it (and no revert of that revert redid it): a
        hand revert, or drive's own before a restart lost its record."""
        by = {}
        for rec in filter(None, (x.strip() for x in _out(self.repo, "log", "--format=%H%x00%B%x1e",
                                                          f"{c}..{self.q.base}").split("\x1e"))):
            h, body = rec.split("\x00", 1)
            by.update(dict.fromkeys(REVERTS_RE.findall(body), h))
        n = 0
        while c in by:
            c, n = by[c], n + 1
        return n % 2 == 1

    def _revert(self, commits: list[str]) -> str:
        """`git revert` each of ``commits`` the landing branch (`Queue.into`: `checked`, never `main` then) hasn't undone already
        (`_undone`) on it, in order (`-m 1` for a merge), and push: "pushed", "undone" (each was undone already), "conflict" (one
        doesn't apply) or "refused" (the push lost a race with another push to it: the next tick tries again).  It stays as it
        was unless "pushed".  Off drive it refuses before anything (`queue.Queue._drive_only`)."""
        into, base = self.q.into, self.q.base
        self.q._drive_only(into, "once", False)
        _git(self.repo, "checkout", "-q", "-B", into, base)
        if not (todo := [c for c in commits if not self._undone(c)]):
            return "undone"
        for c in todo:
            merge = len(_out(self.repo, "rev-list", "--parents", "-n", "1", c).split()) > 2
            r = _git(self.repo, "revert", "--no-edit", *(["-m", "1"] if merge else []), c, check=False, env=IDENTITY)
            if r.returncode:
                _git(self.repo, "revert", "--abort", check=False)
                _git(self.repo, "reset", "-q", "--hard", base)
                return "conflict"
        refused = self.q._remote(self.repo, "push", "-q", "--no-verify", "origin", f"{into}:refs/heads/{into}", ok=None).returncode
        _git(self.repo, "reset", "-q", "--hard", base)
        return "refused" if refused else "pushed"

    def _lock_commits(self, base: str, tip: str, reverted: list[Member]) -> list[str]:
        """The `Lock records regenerated for #N...` commits in ``base``..``tip`` that go with ``reverted``: one
        naming any of them that regenerated a lock in a package a reverted PR's change touches.  Left in place, it rewrites the lean-audit.json the PR's own revert restores, which then conflicts,
        or keeps records of reverted Lean."""
        prs = {x.pr for x in reverted}
        touched = set().union(*(changed(self.repo, f"{x.commit}^1", x.commit) for x in reverted)) if reverted else set()
        out = []
        for rec in filter(None, (x.strip() for x in _out(self.repo, "log", "--format=%H%x00%B%x1e", f"{base}..{tip}").split("\x1e"))):
            c, body = rec.split("\x00", 1)
            named = {int(n) for n in re.findall(r"#(\d+)", body.split("\n")[0])}
            if not body.startswith("Lock records regenerated for ") or not prs & named:
                continue
            m = re.search(r"^Regenerated: (.+)$", body, re.M)
            pkgs = [os.path.dirname(f.strip()) for f in (m[1].split(",") if m else [])]
            if any(under(f, [p]) if p else True for f in touched for p in pkgs):
                out.append(c)
        return out

    def _newest_first(self, base: str, tip: str, commits: list[str]) -> list[str]:
        order = {c: i for i, c in enumerate(_out(self.repo, "rev-list", "--topo-order", f"{base}..{tip}").split())}
        return sorted(set(commits), key=lambda c: order.get(c, -1))

    # -- the after-landing run: one run at a time checks `main`'s tip with every check step.  `drive["postland"]`: {"verified":
    #    the newest `main` whose run passed, "pending": the tips landed since, "run": the run of them, or None, "steps": its
    #    steps, "resend": the next run is the one resend of a run the machine killed, "failed": a tip whose run failed with
    #    nothing named red, or passed without a step, which no run checks again}

    def _ancestor(self, a: str, b: str) -> bool:
        return not _git(self.repo, "merge-base", "--is-ancestor", a, b, check=False).returncode

    def landed_since(self, base: str, tip: str) -> list[Member]:
        """The PRs the queue landed in ``base``..``tip``, oldest first: each merge of a PR onto `next` (`Queue: #N`, with its
        `PR`, `Head` and `Owner` trailers) that `main` reached, but those a revert in that range undid.  Nothing else: never a lander's commit."""
        from .queue import TRAILER_RE

        out, undone = [], set()
        for rec in filter(None, (x.strip() for x in _out(self.repo, "log", "--reverse", "--topo-order", "--format=%H%x00%P%x00%B%x1e",
                                                          f"{base}..{tip}").split("\x1e"))):
            commit, parents, body = rec.split("\x00", 2)
            undone |= set(REVERTS_RE.findall(body))
            t = dict(TRAILER_RE.findall(body))
            if len(parents.split()) > 1 and body.startswith("Queue: #") and {"PR", "Head", "Owner"} <= set(t):
                out.append(Member(int(t["PR"].lstrip("#")), t["Head"], t["Owner"], commit))
        return [x for x in out if x.commit not in undone]

    def follow_postland(self, drive: dict) -> None:
        """The post-landing run, once it ends: passed, its tip is verified; failed, `_postland_failed`; never started (every twin
        exited 75, or was stopped), the next `start_postland` launches it again.  One already failing under `--keep-going` is
        named red at once (`_red`), and cancelled once only its `TAIL` is left.  A pass verifies its tip only when every step it
        ran passed in it (`_unrun`).  One still launching (`_resolve`) runs; one whose train.sh registered no run is told to
        `CI`, and launched again."""
        if not (pl := drive.get("postland")) or not (run := pl.get("run")):
            return
        if run.get("shards") and not self._shards_launched(drive, pl, run):
            return
        if (tail := self._resolve(run, f"the post-landing run of `{self.q.into}` {run['tip'][:12]}")) is not None:
            pl["run"] = None
            if run.get("resend"):
                pl["resend"] = True
            self.refuse(drive, f"launch:postland:{run['tip']}", f"train.sh {run['tip'][:12]} {self.on} --deferred "
                                                                f"{','.join(pl.get('steps') or AFTER_LANDING)} registered no run: {tail}")
            return
        if "launch" in run:
            return
        tip = run["tip"]
        states = self._states(run["ids"])
        if self._lost(run, states):
            if not run.get("resend"):
                pl["run"], pl["resend"] = None, True
                self.q.log(f"research queue drive: the post-landing run {' '.join(run['ids'])} of {tip[:12]} is lost (no "
                           f"state read {LOST_AFTER_S // 3600} h after its launch): it launches again once")
                return
            states = {r: ("lost", -1) for r in states}
        early = next((s[3] for s in states.values() if len(s) > 3 and s[0] in ACTIVE and s[3]), None)
        if not early and any(st in ACTIVE for st, *_ in states.values()):
            self._gate_passed(pl, run)
            return
        if not early and self._every_shard(run, states, lambda rc: rc == 0):
            pl["run"] = None
            self._gate_passed(pl, run)
            if unrun := self._unrun(run["ids"], [s for s in pl.get("steps") or AFTER_LANDING if s not in run.get("skip", ())]):
                if run.get("baseline"):
                    self._red_baseline(drive, pl, run, f"it passed without passing {', '.join(unrun)}")
                    return
                pl["failed"] = tip
                self.refuse(drive, f"postland:{tip}", f"the post-landing run {' '.join(run['ids'])} of `{self.q.into}` {tip[:12]} "
                                                      f"passed without passing {', '.join(unrun)} (skipped, or absent from its "
                                                      "steps.jsonl): it verified nothing and reverted nothing. Find why; "
                                                      f"the next landing's run checks `{self.q.into}` again")
                return
            pl["verified"], pl["verified_run"] = tip, " ".join(run["ids"])
            # a step it skipped by name (`start_postland`) still holds from the run that verified `main` before
            kept = set((pl.get("verified_as") or {}).get("steps") or ()) & set(run.get("skip", ()))
            pl["verified_as"] = {**run["as"], "steps": [s for s in AFTER_LANDING if s in kept | set(run["as"]["steps"])]}
            pl["pending"] = [p for p in pl["pending"] if not self._ancestor(p, tip)]
            pl["landed_at"] = {p: t for p, t in (pl.get("landed_at") or {}).items() if p in pl["pending"]}
            self._red(drive, pl, run, ended=True)
            self.q.log(f"research queue drive: `{self.q.into}` {tip[:12]} verified: its post-landing run {' '.join(run['ids'])} passed "
                       f"({', '.join(pl.get('steps') or ())})" + (f"; {len(pl['pending'])} landing(s) above it wait" if pl["pending"] else ""))
            return
        if not early and self._never_ran(run, states):
            pl["run"] = None
            self.q.log(f"research queue drive: the post-landing run {' '.join(run['ids'])} of {tip[:12]} never ran or was "
                       "stopped: it launches again")
            return
        if early:
            if not run.get("baseline"):
                self._red(drive, pl, run)
            if not self._tail_left(run):
                return
        if early and not (self.dry_run or self.q.shadow):
            self.cancel(self.repo, [r for r, (st, *_) in states.items() if st in ACTIVE],
                        f"drive: the post-landing run of {tip[:12]} is red ({early}) and only {', '.join(TAIL)} are left")
        why = (f"it has already failed, still running: {early}" if early
               else ", ".join(f"{st} rc={rc}" for st, rc, *_ in states.values()))
        # not SIGTERM: a twin a sibling won the slot from, or a cancel (all of them: launched again above)
        signalled = not early and any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC))
                                      for _, rc, *_ in states.values())
        self._gate_passed(pl, run)
        self._postland_failed(drive, pl, run, why, signalled)
        pl["run"] = None

    def _gating(self, commit: str) -> list[str]:
        """The after-landing steps `main` waits for at ``commit`` (`_promote`): every step but those a run skips by name
        (`AFTER_LANDING_OPTIONAL`) and the jobs ``commit``'s own ci.toml takes off the merge gate (`gate = false`:
        circuit-check, `merge.declared_jobs`), which still run there and revert their culprit on `checked`; every such step when
        that ci.toml can't be read."""
        from .merge import declared_jobs

        cache = self.__dict__.setdefault("_gating_of", {})
        if commit not in cache:
            jobs = declared_jobs(self.repo, commit)
            cache[commit] = [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL and (isinstance(jobs, str) or s in jobs)]
        return cache[commit]

    def _gate_passed(self, pl: dict, run: dict) -> None:
        """An after-landing run, ended or still running, that passed every step `main` waits for at its tip (`_gating`),
        whatever a step off the gate does or did (circuit-check, pending or red): its tip is `gated` (with "gated_run", its ids),
        a commit `_promote` may move `main` to."""
        tip, ids = run["tip"], " ".join(run["ids"])
        if (g := pl.get("gated")) and (g == tip or not self._ancestor(g, tip)):
            return
        passed = {st.get("step") for d in self._evidence(Verdict("failed", ids)) for st in step_records(d)
                  if st.get("status") == "passed"}
        if set(self._gating(tip)) <= passed:
            pl["gated"], pl["gated_run"] = tip, ids

    def _shards_launched(self, drive: dict, pl: dict, run: dict) -> bool:
        """Whether every shard of a sharded after-landing run (`start_postland`) has its run ids, which then become the run's own
        (``ids``, ``groups``: each shard's twins).  One whose train.sh registered no run stops the others and is told, as an
        unsharded one is, and the next `start_postland` launches the run again."""
        shards, n = run["shards"], len(run["shards"])
        for k, rec in enumerate(shards, 1):
            if (tail := self._resolve(rec, f"shard {k}/{n} of the post-landing run of `{self.q.into}` {run['tip'][:12]}")) is None:
                continue
            if (got := [r for x in shards for r in x.get("ids") or ()]) and not (self.dry_run or self.q.shadow):
                self.cancel(self.repo, got, f"drive: shard {k}/{n} of the post-landing run of {run['tip'][:12]} registered no run")
            pl["run"] = None
            if run.get("resend"):
                pl["resend"] = True
            self.refuse(drive, f"launch:postland:{run['tip']}", f"train.sh {run['tip'][:12]} {self.on} --deferred "
                                                                f"{','.join(pl.get('steps') or AFTER_LANDING)} --shard {k}/{n} "
                                                                f"registered no run: {tail}")
            return False
        if any("launch" in rec for rec in shards):
            return False
        run["groups"] = [list(rec["ids"]) for rec in shards]
        run["ids"] = [r for g in run["groups"] for r in g]
        del run["shards"]
        return True

    @staticmethod
    def _every_shard(run: dict, states: dict, ok: Callable[[object], bool]) -> bool:
        """Whether in each shard of ``run`` (``groups``; an unsharded run is one) some twin's exit is ``ok``."""
        return all(any(ok(states[r][1]) for r in g if r in states) for g in run.get("groups") or [run["ids"]])

    def _never_ran(self, run: dict, states: dict) -> bool:
        """Whether a shard of ``run`` never ran (each of its twins exited 75 or was stopped) and every other one passed or never
        ran either: the run launches again, whole."""
        never = (NOT_STARTED, SIGTERM_RC)
        groups = run.get("groups") or [run["ids"]]
        missed = [g for g in groups if all(states[r][1] in never for r in g if r in states)]
        return bool(missed) and all(g in missed or any(states[r][1] == 0 for r in g if r in states) for g in groups)

    def _unrun(self, ids: list[str], steps: list[str]) -> list[str]:
        """The ``steps`` no run among ``ids`` records as passed in its steps.jsonl: a run's exit 0 says only that
        nothing failed (a step `skipped:`, or one `--deferred` maps to no part, passes it too)."""
        got = set()
        for d in self.evidence(self.repo, ids):
            with contextlib.suppress(OSError):
                for line in (d / PROGRESS[0]).read_text().splitlines():
                    with contextlib.suppress(ValueError):
                        if (rec := json.loads(line)).get("status") == "passed":
                            got.add(rec.get("step"))
        return [s for s in steps if s not in got]

    def _postland_failed(self, drive: dict, pl: dict, run: dict, why: str, signalled: bool = False) -> None:
        """A post-landing run of ``run["tip"]`` that failed.  Killed by the machine (``signalled``: by a signal or 128+N, or a
        failure that names no file and was a kill: `implicated`, `killed`), it runs again once, on `main`'s tip then, and killed
        again is told to `CI`.  A baseline run's failure is `_red_baseline`'s.  Else each failed suite or step is named red with
        its culprit (`_red`), whose revert waits for its owners' fix (`_red_clock`); a run whose records name none is told to
        `CI`.  The next run checks `main`'s tip."""
        tip, ids, base = run["tip"], " ".join(run["ids"]), pl["verified"]
        members = self.landed_since(base, tip)
        was_killed = signalled or (bool(members) and any(implicated(d) is None and killed(d, why)
                                                         for d in self.evidence(self.repo, run["ids"])))
        since = " ".join(f"#{x.pr}" for x in members) or "none"
        head = (f"the post-landing run {ids} of `{self.q.into}` {tip[:12]} ({', '.join(pl.get('steps') or ())}) failed ({why}); PRs "
                f"landed since {base[:12]}, its last verified `{self.q.into}`: {since}")
        key = f"postland:{tip}"
        if was_killed and not run.get("resend"):
            pl["resend"] = True
            self.q.log(f"research queue drive: {head}. It was killed, a machine's fault: it runs again once, on `{self.q.into}`'s tip")
            return
        if run.get("baseline"):
            self._red_baseline(drive, pl, run, why)
            return
        pl["failed"] = tip
        if was_killed:
            self.refuse(drive, key, f"{head}. Reverted nothing: it was killed again on its resend, a machine's fault")
        elif not self._red(drive, pl, run, ended=True):
            self.refuse(drive, key, f"{head}. Its records name no failed suite or step: reverted nothing. Find why; the "
                                    f"next landing's run checks `{self.q.into}` again")

    def _lost(self, run: dict, states: dict) -> bool:
        """Whether every twin of ``run`` that hasn't ended has had no readable state (`unknown`) for `LOST_AFTER_S` since its
        launch: it is taken as lost, as a killed run is, so a store or node that never answers doesn't hold drive forever."""
        live = [st for st, *_ in states.values() if st in ACTIVE]
        return bool(live) and all(st == "unknown" for st in live) and self.now() - run.get("at", self.now()) > LOST_AFTER_S

    def _red_baseline(self, drive: dict, pl: dict, run: dict, why: str) -> None:
        """The review landing's first run, of `main` itself, failed: `main` was red before any review landing.  It is verified
        anyway with the steps it passed alone (`verified_as`), so landing goes on and a later failure reverts only on a step that
        passed here (`_unconfirmed`); `CI` is told."""
        tip, steps = run["tip"], (run.get("as") or {}).get("steps") or list(AFTER_LANDING)
        unrun = self._unrun(run["ids"], steps)
        pl["verified"], pl["verified_as"] = tip, {"steps": [s for s in steps if s not in unrun], "no_cache": run["as"]["no_cache"]}
        pl["verified_run"] = " ".join(run["ids"])
        pl["pending"] = [p for p in pl["pending"] if not self._ancestor(p, tip)]
        self.refuse(drive, f"postland:{tip}", f"the baseline after-landing run {' '.join(run['ids'])} of `{self.q.into}` {tip[:12]} failed "
                                              f"({why}): `{self.q.into}` was red before any review landing. Landing goes on, and until a "
                                              f"run of `{self.q.into}` passes {', '.join(unrun) or 'them'}, their failure reverts nothing. "
                                              f"Fix `{self.q.into}`")

    def _failing(self, run: dict) -> list[str]:
        """The steps a post-landing run's steps.jsonl records as failed."""
        return sorted({s for d in self.evidence(self.repo, run["ids"]) for st in failed_steps(d) if (s := st.get("step"))})

    def _revert_landed(self, drive: dict, pl: dict, tip: str, base: str, one: Member, head: str, key: str, blame: str) -> bool:
        """Revert ``one``, landed in ``base``..``tip``, on the landing branch with the lock commits that go with it
        (`_lock_commits`), and tell `CI` and its owner.  False when the revert's push lost a race: the next tick tries again."""
        into = self.q.into
        if self.dry_run or self.q.shadow:
            self.q.log(f"research queue drive: would revert #{one.pr} on `{into}`: {head}")
            return True
        locks = self._lock_commits(base, tip, [one])
        commits = self._newest_first(base, tip, [one.commit, *locks])
        got = self._revert(commits)
        if got == "refused":
            self.q.log(f"research queue drive: reverting #{one.pr} lost a race to `{into}`'s push; the next tick tries again")
            return False
        if got == "conflict":
            pl["failed"] = tip
        undo = f"`git revert {' '.join(c[:12] for c in commits)}` (`-m 1` for a merge)"
        with_locks = f", with the lock commit {' '.join(c[:12] for c in locks)} regenerated for it" if locks else ""
        lead = {"pushed": f"#{one.pr} reverted on `{into}`{with_locks}", "undone": f"#{one.pr} was reverted on `{into}` already",
                "conflict": f"#{one.pr} must be reverted by hand ({undo})"}[got]
        self.refuse(drive, key, f"{lead}: {head}. {blame}"
                                + (", and the revert didn't apply" if got == "conflict" else ". Fix and re-mark"), self._owner(one))
        return True

    # -- a red after-landing run (review landing): each failed suite and step is its own record in `postland["red"]`, keyed
    #    `suite NAME` or `step NAME`: {"tip", "ids": the run's, "first": its first error line, "base": where it last passed,
    #    "culprit": [pr, head, owner, commit] or None, "cands": the landings a bisection decides among, "rung": when its
    #    culprit was rung, "fix": what holds its revert, "reverted"}.  `postland["green"]`: {key: the newest tip a run passed it at}

    def _red(self, drive: dict, pl: dict, run: dict, ended: bool = False) -> int:
        """Read an after-landing run's records, still running or ``ended``: each suite and step it passed is green at its tip
        (`_green`), and each it failed is named (`_name`): a step a cancel stopped is `not run`, never failed.  pytest and
        lean-suites are named by their suites, and only as a step when no suite of theirs failed (a timeout, a kill).  The
        number of failed suites and steps."""
        tip = run["tip"]
        dirs = self._evidence(Verdict("failed", " ".join(run["ids"])))
        fails: dict[str, tuple[Path, dict]] = {}
        stopped: set[str] = set()
        for d in dirs:
            for name, s in suite_rows(d).items():
                if s["status"] == "failed":
                    fails[f"suite {name}"] = (d, s)
                elif s["status"] == "passed" and ended:
                    self._green(pl, f"suite {name}", tip)
            for st in step_records(d):
                step = st.get("step") or ""
                if not_run(st):
                    stopped.add(step)
                elif st.get("status") == "failed":
                    if step not in ("pytest", "lean-suites") or not any(s["step"] == step for _, s in fails.values()):
                        fails[f"step {step}"] = (d, st)
                elif st.get("status") == "passed":
                    self._green(pl, f"step {step}", tip)
        if not (self.dry_run or self.q.shadow):
            for key, (d, rec) in fails.items():
                self._name(drive, pl, run, key, d, rec, sorted(stopped))
            self._start_bisect(pl)
        return len(fails)

    def _green(self, pl: dict, key: str, tip: str) -> None:
        """``key`` passed at ``tip``: its newest green, and its red record at or below ``tip`` (and a bisection of it) ends."""
        green = pl.setdefault("green", {})
        if green.get(key) != tip and (key not in green or self._ancestor(green[key], tip)):
            green[key] = tip
        if (r := (pl.get("red") or {}).get(key)) and self._ancestor(r["tip"], tip):
            del pl["red"][key]
            if (pl.get("bisect") or {}).get("key") == key:
                del pl["bisect"]
            self.q.log(f"research queue drive: {key} passed on `{self.q.into}` {tip[:12]}; its red record of {r['tip'][:12]} ends")

    def _base_of(self, pl: dict, key: str, step: str, tip: str) -> str | None:
        """The newest commit at or below ``tip`` an after-landing run passed ``key`` at: its green, else the last verified
        `main` when that run passed ``step`` (`verified_as`), else None."""
        if (g := (pl.get("green") or {}).get(key)) and self._ancestor(g, tip):
            return g
        vs = pl.get("verified_as")
        return pl["verified"] if not vs or step in (vs.get("steps") or ()) else None

    def _own(self, m: Member) -> set[str]:
        """The files ``m``'s merge onto `next` changes, read once per tick."""
        cache = self.__dict__.setdefault("_own_files", {})
        if m.commit not in cache:
            cache[m.commit] = changed(self.repo, f"{m.commit}^1", m.commit)
        return cache[m.commit]

    def _input_rule(self, key: str, d: Path, rec: dict, tip: str) -> tuple[str, Callable[[str], bool]] | None:
        """What failed's declared inputs (`by_inputs`, for one suite or step): a suite's directory and `[tool.verity.tests]
        inputs`, a Lean step's packages or Lean sources, circuit-check's."""
        if key.startswith("suite "):
            ins = suite_inputs(self.repo, tip, rec.get("suite") or "")
            return f"{key}'s declared inputs", lambda f: under(f, ins)
        step = rec.get("step") or ""
        if step.startswith("lean") and step != "lean-suites":
            pkgs = sorted(self._failed_packages(d, rec, tip))
            return (f"Lean package {', '.join(pkgs)}", lambda f: under(f, pkgs)) if pkgs else ("Lean sources", lambda f: f.endswith(LEAN_INPUTS))
        if step == "circuit-check":
            return "circuit-check's inputs", lambda f: forces_cold([f])
        return None

    def _path_owner(self, path: str) -> str | None:
        """The handle whose `owns` in `main`'s registry is ``path``'s longest prefix (`research msg owners`)."""
        from .queue import REGISTRY
        from .slack import owner

        if (reg := self.__dict__.get("_reg")) is None:
            r = _git(self.repo, "show", f"origin/main:{REGISTRY}", check=False)
            try:
                reg = json.loads(r.stdout) if not r.returncode else {}
            except ValueError:
                reg = {}
            self._reg = reg = reg if isinstance(reg, dict) and isinstance(reg.get("handles"), list) else {"handles": []}
        try:
            return owner(reg, path)
        except (KeyError, TypeError, AttributeError):
            return None

    def _name(self, drive: dict, pl: dict, run: dict, key: str, d: Path, rec: dict, stopped: list[str]) -> None:
        """Name the culprit of failed ``key`` among the PRs landed since it last passed (`_base_of`): the one whose change
        touches a file the failure points at (its failing tests, the files its log names, what the read guard caught it reading
        and the test modules that read it), else the one that touches its declared inputs (`_input_rule`); for ratchet-gate,
        the oldest whose merge the gate's log says it fails (`ratchet_failures`) before either; several, a bisection
        by landing decides (`_start_bisect`).  Ring @ci, the suite's owner and the culprit's (`_ring`): drive reverts the
        culprit `FIX_WINDOW_S` after unless a fix is marked (`_red_clock`).  A key already named for this tip, or whose culprit
        is still on `main` unreverted, is left as it is."""
        red, tip = pl.setdefault("red", {}), run["tip"]
        if (old := red.get(key)) and (old["tip"] == tip or (pl.get("bisect") or {}).get("key") == key
                                      or (old.get("culprit") and not old.get("reverted") and not self._undone(old["culprit"][3]))):
            if old["tip"] != tip:
                old["seen"] = tip                       # red again here: a fix that landed below this tip didn't fix it (`_fix`)
            return
        step = rec.get("step") or ""
        r = {"tip": tip, "ids": run["ids"], "first": first_error(d, rec), "step": step, "culprit": None,
             "owner": self._path_owner(rec["suite"]) if rec.get("suite") else None, "not_run": stopped}
        red[key] = r
        gate = ratchet_failures(d, rec) if step == RATCHET_STEP else []
        if gate:
            r["first"] = f"`{gate[0][0]} ({gate[0][1]}): failed: {gate[0][2][:200]}`"
        # the gate judges every merge above the landed branch's verified tip and names the one it fails, wherever the step last passed
        if (base := self._base_of(pl, key, step, tip) or (pl.get("verified") if gate else None)) is None:
            self._ring(drive, r, key, f"it never passed in an after-landing run since `{self.q.into}` {pl['verified'][:12]}, so no "
                                      "landing can be named. Reverted nothing")
            return
        r["base"] = base
        members = self.landed_since(base, tip)
        paths = suite_paths(d, rec) if key.startswith("suite ") else step_paths(d, rec)
        hits = [x for x in members if touches(self._own(x), paths)] if paths else []
        why = f"its change touches {', '.join(sorted({f for x in hits for f in touches(self._own(x), paths)})[:4])}"
        r["paths"] = sorted(paths or ()) + sorted({f for x in hits for f in touches(self._own(x), paths)})
        if gate and (named := [x for x in members if x.commit[:12] in {c for _, c, _ in gate}][:1]):
            hits, why = named, "the ratchet gate fails its merge"
        if not hits and (rule := self._input_rule(key, d, rec, tip)):
            label, ok = rule
            hits, why = [x for x in members if any(ok(f) for f in self._own(x))], f"its change touches {label}"
            r["paths"] += sorted({f for x in hits for f in self._own(x) if ok(f)})
        since = f"{len(members)} landing(s) since `{self.q.into}` {base[:12]}, where it last passed"
        if len(hits) == 1:
            r.update(culprit=[hits[0].pr, hits[0].head, hits[0].owner, hits[0].commit], why=why, rung=self.now())
            self._ring(drive, r, key, f"Culprit: #{hits[0].pr} ({why}), of the {since}")
        elif hits and REVERTS_RE.search(_out(self.repo, "log", "--format=%B", f"{base}..{tip}")):
            self._ring(drive, r, key, f"{' '.join(f'#{x.pr}' for x in hits)} each touch it ({since}), and a revert lies in "
                                      f"{base[:12]}..{tip[:12]}, so bisecting could blame the wrong PR. Reverted nothing; find "
                                      "the culprit by hand")
        elif hits:
            r["cands"] = [[x.pr, x.head, x.owner, x.commit] for x in hits]
            self._ring(drive, r, key, f"{' '.join(f'#{x.pr}' for x in hits)} each touch it ({since}): drive bisects them by "
                                      "landing, and rings the one it finds")
        else:
            self._ring(drive, r, key, f"none of the {since} touches it or its declared inputs, so it was red before them or the "
                                      "machine failed. Reverted nothing")

    def _ring(self, drive: dict, r: dict, key: str, what: str, tag: str = "") -> None:
        """Ring @ci, the owner of ``key``'s suite and its culprit's owner with what failed, the run, its first error line and
        what drive does next."""
        c = r.get("culprit")
        owner = self._owner(Member(*c)) if c else None
        nr = f" Not run (a cancel stopped them): {', '.join(r['not_run'])}." if r.get("not_run") else ""
        clock = ("" if not c else
                 f" Drive reverts #{c[0]} now, and lands no PR touching circuit-check's inputs until it passes on "
                 f"`{self.q.into}`" if key == CIRCUIT_RED else
                 f" Drive reverts #{c[0]} at {time.strftime('%H:%MZ', time.gmtime(r['rung'] + FIX_WINDOW_S))} unless a fix is "
                 f"marked by then: a ready PR whose body names #{c[0]} or says `fixes {key}`, or whose change touches what "
                 f"failed, or a new head of #{c[0]}")
        self.refuse(drive, f"red:{key}:{c[3] if c else r['tip']}{tag}",
                    f"{key} is red on `{self.q.into}` {r['tip'][:12]} (run {' '.join(r['ids'])}): {r['first']}. {what[:1].upper()}{what[1:]}."
                    f"{clock}{nr}",
                    r.get("owner"), owner)

    def _start_bisect(self, pl: dict) -> None:
        """Bisect the first red record with candidates and no culprit (`_bisect_step`), when no bisection runs: its failed
        suite (or step) alone decides each landing, with the step that runs it."""
        if pl.get("bisect"):
            return
        for key, r in (pl.get("red") or {}).items():
            if r.get("cands") and not r.get("culprit") and not r.get("bisected"):
                cands = r["cands"]
                r["bisected"] = True
                pl["bisect"] = {"tip": r["tip"], "base": r["base"], "steps": [r["step"]], "why": r["first"], "key": key,
                                "suite": key.split(" ", 1)[1] if key.startswith("suite ") else None,
                                "no_cache": self.after_landing_no_cache, "cands": cands, "good": -1,
                                "bad": len(cands) - 1 if cands[-1][3] == r["tip"] else len(cands), "run": None}
                self.q.log(f"research queue drive: bisecting {key}, red on `{self.q.into}` {r['tip'][:12]}, among "
                           f"{' '.join(f'#{c[0]}' for c in cands)}")
                return

    def _suite_status(self, ids: list[str], suite: str) -> str | None:
        """``suite``'s status in the reports of run ``ids``: passed, failed, or None when none names it."""
        got = {s["status"] for d in self.evidence(self.repo, ids) for n, s in suite_rows(d).items() if n == suite}
        return "failed" if "failed" in got else "passed" if "passed" in got else None

    def _red_clock(self, drive: dict) -> None:
        """Revert each red record's culprit `FIX_WINDOW_S` after its ring, unless a fix is marked (`_fix`) or `main` undid it
        already: `git revert -m 1` of its merge onto `next` (`_revert_landed`), its owner told.  A red circuit-check's culprit
        (`CIRCUIT_RED`) is reverted at once, a marked fix or not."""
        pl = drive.get("postland") or {}
        if self.dry_run or self.q.shadow:
            return
        for key, r in list((pl.get("red") or {}).items()):
            now = key == CIRCUIT_RED
            if not (c := r.get("culprit")) or r.get("reverted") or not now and self.now() - r["rung"] < FIX_WINDOW_S:
                continue
            one = Member(*c)
            if self._undone(one.commit):
                r["reverted"] = "undone"
                continue
            if not now and (fix := self._fix(key, r)):
                if r.get("fix") != fix:
                    r["fix"] = fix
                    self.q.log(f"research queue drive: not reverting #{one.pr} for {key}: a fix is marked, {fix}")
                continue
            r.pop("fix", None)
            head = f"{key} is red on `{self.q.into}` {r['tip'][:12]} (run {' '.join(r['ids'])}): {r['first']}"
            blame = (f"#{one.pr} is its culprit ({r['why']}), and a red circuit-check is reverted at once" if now else
                     f"#{one.pr} is its culprit ({r['why']}), and no fix was marked within {FIX_WINDOW_S // 60} min of the "
                     "ring")
            if self._revert_landed(drive, pl, r["tip"], r["base"], one, head, f"revert:{key}:{one.commit}", blame):
                r["reverted"] = self.now()

    def _fix(self, key: str, r: dict) -> str | None:
        """A marked fix of red ``key``: an open PR marked ready at its head that is a new head of the culprit, or whose body names
        the culprit (`#N`) or says `fixes suite NAME` (case aside), or whose diff touches a path the failure named (`r["paths"]`,
        those the culprit was named by).  A body that merely mentions the suite doesn't count: `research` or `check` is in
        almost any body, and would hold the revert for good.  A fix that landed on `main` after the last tip a run found it red
        at (`seen`, else the record's tip) counts too, by its merge's title or its change: once landed it is no open PR, and
        drive would otherwise ask for a revert of a culprit whose fix already landed.  None when there is none."""
        pr, head = r["culprit"][0], r["culprit"][1]
        name = key.split(" ", 1)[1]
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        says = re.compile(rf"#{pr}\b|\bfixes\s+(?:suite|step)\s+`?{re.escape(name)}(?![\w/.-])", re.I)
        for p in sorted(self._prs.values(), key=lambda p: p.number):
            if p.number == pr and p.head == head or not self.q.ready(p.number, p.head):
                continue
            if p.number == pr:
                return f"#{p.number} at {p.head[:8]}, a new head of the culprit"
            if says.search(p.body or ""):
                return f"#{p.number} at {p.head[:8]}"
            if r.get("paths") and (hit := touches(self._pr_files(p), set(r["paths"]))):
                return f"#{p.number} at {p.head[:8]}, whose change touches {', '.join(sorted(hit)[:3])}"
        main, since = self.q.base, r.get("seen") or r["tip"]
        if not self._ancestor(since, main):
            return None
        for m in self.landed_since(since, main):
            if m.pr == pr:
                continue
            if says.search(_out(self.repo, "log", "-1", "--format=%s", m.commit)):
                return f"#{m.pr}, landed at {m.commit[:8]}"
            if r.get("paths") and (hit := touches(changed(self.repo, f"{m.commit}^1", m.commit), set(r["paths"]))):
                return f"#{m.pr}, landed at {m.commit[:8]}, whose change touches {', '.join(sorted(hit)[:3])}"
        return None

    def _pr_files(self, p) -> set[str]:
        """The files an open PR's head changes against `main` (from their merge base), read once per tick; none when its head
        can't be fetched."""
        cache = self.__dict__.setdefault("_own_files", {})
        if (k := f"pr:{p.head}") not in cache:
            if _git(self.repo, "cat-file", "-e", f"{p.head}^{{commit}}", check=False).returncode:
                _git(self.repo, "fetch", "-q", "origin", f"+refs/pull/{p.number}/head:refs/queue/fix/{p.number}", check=False)
            mb = _git(self.repo, "merge-base", "origin/main", p.head, check=False)
            cache[k] = changed(self.repo, mb.stdout.strip(), p.head) if not mb.returncode and mb.stdout.strip() else set()
        return cache[k]

    def _tail_left(self, run: dict) -> bool:
        """Whether every step of an after-landing run but `TAIL` (and those it skips by name) has ended, by its steps.jsonl, and,
        while no step `main` waits for (`_gating`) failed, those of `TAIL` too: a red only off the gate (circuit-check) cancels
        nothing that could still let `_promote` move `main` to its tip."""
        recs = [st for d in self._evidence(Verdict("failed", " ".join(run["ids"]))) for st in step_records(d)]
        ended = {st.get("step") for st in recs}
        gating = self._gating(run["tip"])
        held = any(st.get("step") in gating and st.get("status") == "failed" for st in recs)
        return all(s in ended for s in AFTER_LANDING
                   if s not in run.get("skip", ()) and (s not in TAIL or (not held and s in gating)))

    def _behind(self, drive: dict) -> None:
        """Ring @ci and @top, every `BEHIND_EVERY_S`, while the oldest landing no after-landing run has verified landed more
        than `BEHIND_S` ago, unless a run that started under `POSTLAND_RUN_S` ago checks `main` with nothing red: then
        verification is only under way (a cold run is ~4 h).  Drive's tick, not top's hourly sweep, rings it: drive holds the
        last verified `main`, its landings since and what is red."""
        if not (pl := drive.get("postland")) or self.dry_run or self.q.shadow:
            return
        main = _out(self.repo, "rev-parse", self.q.base)
        pending = [p for p in pl.get("pending") or () if not self._ancestor(p, pl["verified"])]
        if not pending or main == pl["verified"]:
            pl.pop("rang_behind", None)
            return
        at = pl.setdefault("landed_at", {})
        since = min(at.get(p) or int(_out(self.repo, "log", "-1", "--format=%ct", p)) for p in pending)
        run, red, now = pl.get("run"), pl.get("red") or {}, self.now()
        if now - since < BEHIND_S or (run and not red and now - run.get("at", now) < POSTLAND_RUN_S):
            return
        if now - pl.get("rang_behind", 0) < BEHIND_EVERY_S:
            return
        reds = "; ".join(f"{k}" + (f" (culprit #{r['culprit'][0]}" + (", reverted)" if r.get("reverted") else ")")
                                   if r.get("culprit") else "") for k, r in red.items()) or "nothing named red"
        state = (f"its run {_ids(run)} of {run['tip'][:12]} has run {(now - run.get('at', now)) / 3600:.1f} h" if run else
                 "no after-landing run is going")
        text = (f"`{self.q.into}`'s last verified commit {pl['verified'][:12]} is {(now - since) / 3600:.1f} h behind its tip {main[:12]} "
                f"({len(pending)} landing(s) unverified); {state}; red: {reds}")
        if self.post(f"{CI} {TOP} queue: {text}", (CI, TOP)):
            pl["rang_behind"] = now
            self.q.log(f"research queue drive: rang {CI} {TOP}: {text}")

    def start_postland(self, drive: dict) -> None:
        """Launch the one post-landing run, on the landing branch's tip (`Queue.into`), when none runs, a landing awaits it, and
        that tip is neither verified nor one whose run failed with nothing drive could revert."""
        if not (pl := drive.get("postland")) or pl.get("run"):
            return
        q = self.q
        main = _out(self.repo, "rev-parse", q.base)
        pl["pending"] = [p for p in pl["pending"] if not self._ancestor(p, pl["verified"])]
        base = not self._baselined(pl)                            # the first run: `main` itself (`_unbaselined`)
        if not base and (not pl["pending"] or main == pl.get("failed") or main == pl["verified"]):
            return
        pl["steps"] = list(AFTER_LANDING)
        steps = ",".join(pl["steps"])
        # it sends the upstream build when the landings since the last verified `main` change what lean-agreement checks (as
        # check.py finds against that base); lean-changed has no cache to compare with under --no-cache;
        # lean-audit fails on a record difference, since nothing commits a regenerated record after landing; circuit-check
        # reuses its per-target passes all the same, since a cold `--all` (~4 h) outlasts an after-landing run
        from .merge import required_steps

        extra = {"agreement": base or "lean-agreement" in required_steps(self.repo, pl["verified"], main, "check"),
                 "no_cache": self.after_landing_no_cache, "circuit_cache": self.after_landing_no_cache, "no_regenerate": True}
        skip = [s for s, off in (("lean-agreement", not extra["agreement"]), ("lean-changed", extra["no_cache"])) if off]
        # ratchet-gate judges the landings above the run's base and fails one at or above the tip: the baseline's tip is the
        # verified `main` itself, so its base is the tip's first parent
        below = (pl["verified"] if pl["verified"] != main else
                 _git(self.repo, "rev-parse", "--verify", "-q", f"{main}^1", check=False).stdout.strip()) or None
        flags = (f" --base {below[:12]}" if below else "") + "".join(f" --{k.replace('_', '-')}" for k, on in extra.items() if on)
        n = self.after_landing_shards
        if self.dry_run or q.shadow:
            q.log(f"research queue drive: would launch train.sh {main[:12]} {self.on} --deferred {steps}{flags}"
                  + (f" as {n} shards" if n > 1 else ""))
            return
        if n == 1:
            rec, tail = self._started(self.launch_postland(self.repo, main, self.on, steps, base=below, **extra))
        else:
            got = [self._started(self.launch_postland(self.repo, main, self.on, steps, base=below, **extra, shard=f"{k}/{n}"))
                   for k in range(1, n + 1)]
            rec, tail = {"ids": [], "shards": [r for r, _ in got]}, next((t for r, t in got if r is None), "")
            if any(r is None for r, _ in got):
                if ids := [i for r, _ in got if r for i in r.get("ids") or ()]:
                    self.cancel(self.repo, ids, f"drive: a shard of the post-landing run of {main[:12]} registered no run")
                rec = None
        if rec is None:
            self.refuse(drive, f"launch:postland:{main}", f"train.sh {main[:12]} {self.on} --deferred {steps}{flags}"
                                                          + (f" (as {n} shards)" if n > 1 else "") + f" registered no run: {tail}")
            return
        pl["run"] = {**rec, "tip": main, "at": self.now(), **({"resend": True} if pl.pop("resend", False) else {}),
                     **({"skip": skip} if skip else {}), **({"baseline": True} if base else {}),
                     "as": {"steps": [s for s in AFTER_LANDING if s not in skip], "no_cache": extra["no_cache"]}}
        q.log(f"research queue drive: post-landing run of `{self.q.into}` {main[:12]} ({steps}): {_ids(rec)}; "
              f"{len(pl['pending'])} landing(s) since {pl['verified'][:12]}")

    def _owner(self, m: Member) -> str | None:
        """The owning lane of a PR the queue landed: its open PR's (`_handle`), else the one its merge's `Owner` names."""
        from .queue import PR

        return self._handle(m.pr) or self.q.handle(PR(m.pr, m.head, body=f"Lane {m.owner}"))

    # -- landing

    def _chain(self) -> tuple[str, list[Member]]:
        """(the base `next` was built on, its members oldest first), whether or not the landing branch has moved since."""
        return built_on(self.repo, f"refs/remotes/origin/{self.q.branch}", self.q.base)

    def _land(self, drive: dict, m: Member, members: list[Member], prs: str, v: Verdict, main: str) -> Member | None:
        """Fast-forward the landing branch (`Queue.into`: `checked`, or `main`) to ``m``, the top of `next`'s reviewed prefix
        (`land_reviewed`).  The after-steps (`_stacked`, `merge.caught_up`, the sweep) are a landing on `main`'s, so on
        `checked` they wait for `_promote`.  Off drive it refuses before anything (`queue.Queue._drive_only`)."""
        q = self.q
        into, base = q.into, q.base
        q._drive_only(into, "once", False)
        _git(self.repo, "checkout", "-q", "-B", into, base)
        r = _git(self.repo, "merge", "-q", "--ff-only", m.commit, check=False)
        if r.returncode:
            _git(self.repo, "merge", "--abort", check=False)
            _git(self.repo, "reset", "-q", "--hard", base)
            self.refuse(drive, f"land:{m.commit}", f"{prs} passed ({v.why}) but doesn't merge onto `{into}` {main[:12]}: "
                                                    f"{(r.stderr or r.stdout).strip()[-200:]}")
            return None
        p = q._remote(self.repo, "push", "-q", "--no-verify", "origin", f"{into}:refs/heads/{into}", ok=None)
        if p.returncode:
            _git(self.repo, "reset", "-q", "--hard", base)
            q.log(f"research queue drive: pushing `{into}` was refused ({p.stderr.strip()[-200:]}); the next tick tries again")
            return None
        tip = _out(self.repo, "rev-parse", "HEAD")
        q.log(f"research queue drive: landed {prs} on `{into}` at {tip[:12]} ({v.why})")
        self._after_landing(drive, tip, main)
        for x in members:
            if (now := self._head(x.pr)) and now != x.head:
                self.refuse(drive, f"stale:{x.pr}@{x.head}", f"#{x.pr} landed at {x.head[:12]}, the head on `{q.branch}`, not "
                                                             f"at {now[:12]}, its head now, so {now[:12]} rejoins `{q.branch}` "
                                                             "above and lands later", self._owner(x))
        if into == "main":
            self._on_main(drive, members, main, tip, v.attempt)
        return m

    def _on_main(self, drive: dict, members: list[Member], since: str, tip: str, attempt: str) -> None:
        """What follows ``members`` reaching `main` at ``tip`` from ``since`` (a landing, or `_promote`): their owners' stacked
        PRs told (`_stacked`), `merge.caught_up` with ``attempt`` as each one's run, and the PR sweep."""
        from .merge import _sweep_after, caught_up

        q = self.q
        self._stacked(drive, members, tip)

        def after() -> None:
            with authed(q.auth):
                caught_up(self.repo, "origin", "check", since=since, log=q.log,
                          runs={x.commit: attempt for x in members} | {tip: attempt})

        # the landing's after-steps (spec DM, mirror, post_train, posts, labels) wait for the tick's launches (a landing tick
        # would hold the next launches behind them); the sweep stays here, since it closes what the sync must not read
        if (pending := getattr(self, "_after", None)) is not None:
            pending.append(after)
        else:
            after()
        with authed(q.auth):
            _sweep_after(self.repo, "main", "origin")

    def _promotable(self, pl: dict) -> list[tuple[str, str]]:
        """The commits of `checked` `main` may move to, each with the run that passed it ("" for ci's baseline): `verified` when
        its steps (`verified_as`: its run's, ci's baseline, `--steps` too, or a red baseline's) cover every step `main` waits
        for there (`_gating`), and `gated`, whose run passed those though a step off the gate (circuit-check) was red."""
        out = []
        if (v := pl.get("verified")) and set(self._gating(v)) <= set((pl.get("verified_as") or {}).get("steps") or ()):
            out.append((v, pl.get("verified_run") or ""))
        if g := pl.get("gated"):
            out.append((g, pl.get("gated_run") or ""))
        return out

    def _promote(self, drive: dict) -> None:
        """When drive lands on `checked` (drive.toml's `land_into`), fast-forward `main` to the newest commit of `checked` whose
        after-landing run passed every step on the merge gate (`_promotable`), and run what a landing on `main` runs for what it
        brings (`_on_main`).  A step off the gate (circuit-check) holds nothing here: its red reverts its culprit on `checked`,
        and `main` follows the revert.  `main` never moves to a commit no such run passed, and never back; one that isn't below
        it (pushed outside drive) waits for `_absorb`."""
        q = self.q
        pl = drive.get("postland") or {}
        if q.into == "main" or self.dry_run or q.shadow:
            return
        main = _out(self.repo, "rev-parse", "refs/remotes/origin/main")
        ok = [(c, r) for c, r in self._promotable(pl)
              if not self._ancestor(c, main) and self._ancestor(main, c) and self._ancestor(c, q.base)]
        if not ok:
            return
        v, run = max(ok, key=lambda cr: sum(self._ancestor(o, cr[0]) for o, _ in ok))
        q._drive_only("main", "once", False)
        p = q._remote(self.repo, "push", "-q", "--no-verify", "origin", f"{v}:refs/heads/main", ok=None)
        if p.returncode:
            q.log(f"research queue drive: moving `main` to {v[:12]} was refused ({p.stderr.strip()[-200:]}); the next tick "
                  "tries again")
            return
        members = self.landed_since(main, v)
        q.log(f"research queue drive: `main` fast-forwarded to {v[:12]}, which its after-landing run {run or '(ci baseline)'} "
              f"passed on every step on the merge gate: {' '.join(f'#{x.pr}' for x in members) or 'no PR'} since {main[:12]}")
        self._on_main(drive, members, main, v, run)

    def _absorb(self, drive: dict) -> None:
        """When drive lands on `checked`, bring into it a `main` that moved outside drive (the lander's pure move, `research
        merge --push` on its own passing check): a fast-forward when `checked` is below it, else a merge of it onto `checked`,
        which the next after-landing run checks with what landed (a conflict is told to `CI`).  Then `_promote` can move `main`
        on again, and the next sync rebuilds `next` on the new `checked`."""
        q = self.q
        if q.into == "main" or self.dry_run or q.shadow:
            return
        main, tip = _out(self.repo, "rev-parse", "refs/remotes/origin/main"), _out(self.repo, "rev-parse", q.base)
        if self._ancestor(main, tip):
            return
        q._drive_only(q.into, "once", False)
        _git(self.repo, "checkout", "-q", "-B", q.into, q.base)
        r = _git(self.repo, "merge", "-q", "--no-edit", *(() if self._ancestor(tip, main) else ("--no-ff", "-m",
                 f"Merge main {main[:12]} into {q.into}")), "refs/remotes/origin/main", check=False, env=IDENTITY)
        if r.returncode:
            _git(self.repo, "merge", "--abort", check=False)
            _git(self.repo, "reset", "-q", "--hard", q.base)
            self.refuse(drive, f"absorb:{main}", f"`main` {main[:12]} moved outside drive and doesn't merge onto `{q.into}` "
                                                 f"{tip[:12]} ({(r.stderr or r.stdout).strip()[-200:]}): `main` stays there until "
                                                 f"`{q.into}` holds it. Merge it by hand on the control pod")
            return
        new = _out(self.repo, "rev-parse", "HEAD")
        p = q._remote(self.repo, "push", "-q", "--no-verify", "origin", f"{q.into}:refs/heads/{q.into}", ok=None)
        _git(self.repo, "reset", "-q", "--hard", q.base)
        if p.returncode:
            q.log(f"research queue drive: pushing `{q.into}` with `main` {main[:12]} was refused ({p.stderr.strip()[-200:]}); "
                  "the next tick tries again")
            return
        if new != main:
            self._after_landing(drive, new, tip)
        q.log(f"research queue drive: `main` {main[:12]} moved outside drive; `{q.into}` "
              + ("fast-forwarded to it" if new == main else f"merged it at {new[:12]}, which the after-landing run checks"))

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
        else:
            self.q.log(f"research queue drive: could not post to the merge thread (the next tick retries): {' '.join(to)}: {text}")

    def tell_contained(self, drive: dict, entries: dict) -> None:
        """Tell each PR marked ready that the sync left off `next` because it contains the head, or the commits at an older
        head, of an open PR that isn't admitted (`Queue.entries`), and that PR, once per pair of heads: else the marked one waits
        and no one hears.  Both owners are rung.  One unmarked at its head is asked for its mark; one ejected, held, without
        an owner or waiting itself is named with why, and the fix: admit it, drop its commits, or have it closed.  A PR that
        isn't marked, or waits for a reason of its own (a hold, a warming check), tells no one; one left out because a PR it
        contains didn't merge is `_tell_out`'s."""
        q = self.q
        hold = q.rules().hold_label
        for n, e in entries.items():
            if e.state != "waiting" or not e.why.endswith("which isn't admitted") or not q.ready(n, e.pr.head):
                continue
            for b in (entries[a] for a in e.after if entries[a].state in ("waiting", "refused")):
                m, h = b.pr.number, b.pr.head
                marked = q.ready(m, h)
                if b.state == "waiting" and b.pr.owner is not None and hold not in b.pr.labels and not marked:
                    if (key := f"unmarked:{n}@{e.pr.head}:{m}@{h}") in drive["told"]:
                        continue
                    by = self.ready_by(n, e.pr.head) or q.handle(e.pr) or e.pr.owner
                    at = _out(q.workdir, "merge-base", e.pr.head, h)
                    self.refuse(drive, key,
                                f"#{n} (marked by {by}) waits for your mark of #{m}, which it contains (at {at[:12]}): "
                                f"mark #{m} ready at {h[:8]}, or withdraw it (`research queue eject {m} --reason ...`) and "
                                f"#{n} carries it", q.handle(b.pr), q.handle(e.pr))
                    continue
                admit = (f"lift #{m}'s `{hold}` label" if hold in b.pr.labels else
                         f"clear what #{m} waits for" if marked and b.state == "waiting" else f"mark #{m} ready at {h[:8]}")
                self.refuse(drive, f"contained:{n}@{e.pr.head}:{m}@{h}",
                            f"#{n} at {e.pr.head[:8]} is out of `{q.branch}`: it contains {e.contains(m)}, and #{m} at "
                            f"{h[:8]} isn't admitted ({b.why}). Fix: {admit}, drop #{m}'s commits from #{n}, or ask {TOP} to "
                            f"close #{m}", q.handle(e.pr), q.handle(b.pr))

    def _tell_out(self, drive: dict, state: dict) -> None:
        """Tell each PR the last sync left off `next` once per head: one that didn't merge (`conflicts`), and one that contains
        such a PR (`waiting`)."""
        for kind, key in (("conflict", "conflicts"), ("waiting", "waiting")):
            for n, c in state.get(key, {}).items():
                self.refuse(drive, f"{kind}:{n}@{c['head']}", f"#{n} at {c['head'][:8]} is out of `{self.q.branch}`: {c['why']}",
                            c.get("handle"))

    def _handle(self, pr: int) -> str | None:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self.q.handle(self._prs[pr]) if pr in self._prs else None

    def _head(self, pr: int) -> str | None:
        """Open PR ``pr``'s head now, None when it isn't open."""
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self._prs[pr].head if pr in self._prs else None

    def _failed_packages(self, run_dir: Path, step: dict, tip: str) -> set[str]:
        """The Lake packages a failed Lean step's log reports `FAIL` for, with each package its manifest requires by path, at
        ``tip``; empty when the log names none (a fallback on every Lean source would blame every PR that touches Lean for one
        package's failure)."""
        try:
            log = (run_dir / Path(step.get("log") or "").name).read_text(errors="replace") if step.get("log") else ""
        except OSError:
            return set()
        out = set()
        for pkg in set(LEAN_AUDIT_RE.findall(log)):
            out.add(pkg)
            r = _git(self.repo, "show", f"{tip}:{pkg}/lake-manifest.json", check=False)
            with contextlib.suppress(ValueError, AttributeError):
                for dep in json.loads(r.stdout).get("packages", []) if not r.returncode else []:
                    if dep.get("type") == "path":
                        out.add(os.path.normpath(os.path.join(pkg, dep.get("dir", ""))))
        return {p for p in out if not p.startswith("..")}

    def _eject(self, drive: dict, key: str, head: str, m: Member, prs: str, v: Verdict, xs: list[Member], because: str) -> None:
        """Eject ``xs`` from ``m``'s failed prefix of `next`, and tell `CI` and their owners why (``because``).  A PR pushed since the head
        its member merged is not ejected: `eject` marks the PR's current head, often the owner's fix, while the old one is still
        on `next`; the next sync checks the new head in its place."""
        ejected, why, pushed = [], because, []
        for x in xs:
            if (now := self._head(x.pr)) not in (None, x.head):
                pushed.append(f"#{x.pr} was pushed to {now[:8]} since {x.head[:8]}, the head that failed: that head isn't "
                              "ejected, and the next sync checks it in its place")
                self.q.log(f"research queue drive: not ejecting #{x.pr}: {pushed[-1]}")
                continue
            reason = f"`{self.q.branch}` {m.commit[:12]} ({prs}), check {v.attempt}: #{x.pr}: {because}"
            if self.q.shadow or self.dry_run:
                self.q.log(f"research queue drive: would eject #{x.pr}: {reason}")
                continue
            try:
                self.q.eject(x.pr, reason, "ci")
                ejected.append(x)
            except QueueError as e:
                why = f"{because}; ejecting #{x.pr} failed: {e}"
        also = "".join(f". {p}" for p in pushed)
        if ejected:
            names, one = " ".join(f"#{x.pr}" for x in ejected), len(ejected) == 1
            self.refuse(drive, key, f"{head}. Ejected {names}: {because}; the next sync rebuilds without "
                                    f"{'it' if one else 'them'}. Fix and re-mark{also}", *(self._handle(x.pr) for x in xs))
            return
        if pushed and why == because:
            self.refuse(drive, key, f"{head}. The cause: {because}{also}", *(self._handle(x.pr) for x in xs))
            return
        self.refuse(drive, key, f"{head}. Not ejected: {why}. Eject the PR that broke it (`research queue eject N --reason ...`); "
                                f"the next sync rebuilds without it{also}", *(self._handle(x.pr) for x in xs))

    # -- the review landing (the module's docstring).  `drive["postland"]` also keeps "verified_as" ({"steps", "no_cache"} of the run that verified `verified`, or ci's
    #    `research queue baseline`), "verified_run" (the ids of the run that verified it; none for ci's baseline), "gated" and
    #    "gated_run" (the newest tip whose run passed every step on the merge gate, red off it or not: `_gate_passed`) and
    #    "bisect" (a failed after-landing run's implicated landings, being bisected: `_bisect_step`)

    def _review(self, state: dict, drive: dict, dry_run: bool) -> str:
        """One tick of the review landing: move `main` to a fully checked `checked` (`_promote`) and bring a `main` pushed
        outside drive into `checked` (`_absorb`), sync `next` (each PR it merges entering once its preflight passed there:
        `queue.Queue.preflight`), land its reviewed prefix (`land_reviewed`) once the landing branch has a baseline
        (`_unbaselined`), and launch the after-landing run, or the next run of a bisection (`_bisect_step`)."""
        q = self.q
        what: list[str] = []
        drive["tick"] = drive.get("tick", 0) + 1
        self._bisect_step(drive)
        self._red_clock(drive)
        self._promote(drive)
        self._absorb(drive)
        q.lift = self._lift(q.chain())
        entries = q.sync(dry_run)[1]
        state = {**q.load_state(), "drive": drive}
        self._tell_out(drive, state)
        self.tell_contained(drive, entries)
        landed, why = None, self._unbaselined(drive)
        if why is None:
            landed, why = self.land_reviewed(drive, dry_run)
        what.insert(0, f"landed #{landed.pr}" if landed else why)
        self._sweep(drive, dry_run)
        self._stale(drive, dry_run)
        self.start_postland(drive)
        self._behind(drive)
        if not dry_run:
            state["ejected"] = {**state.get("ejected", {}), **q.load_state().get("ejected", {})}
            q.save_state(state)
        q.log(f"research queue drive: {'; '.join(what)}")
        return "; ".join(what)

    def _lift(self, chain: list[Member]) -> list[int]:
        """The members the sync moves to the bottom of `next`, in this order (`queue.Queue.lift`, which keeps each after the PRs
        it contains): those with their ready mark and a land verdict at their head (`_reviewed`'s rule), below the members
        waiting for one or for circuit-check to pass (`_circuit_held`) (drive.toml's `reviewed_first`: one waiting for its
        review would hold every reviewed member above it), so they land this tick.  Nothing moves past a held member (a hold
        is a verdict, and keeps what is above it), nor a member whose labels can't be read."""
        if not self.q.flag("reviewed_first"):
            return []
        if self.q.prefetch is not None:
            self.q.prefetch([(m.pr, m.head) for m in chain])   # one pull of every member's labels, not three per member
        landing: list[tuple[Member, bool]] = []                 # each member below the first one nothing moves past
        for m in chain:
            try:
                why = unreviewed(self.reviews(m.pr, m.head), self._owners(m)) if self.q.ready(m.pr, m.head) else "withdrawn"
            except QueueError as e:                             # its verdict unknown, nothing moves past it
                self.q.log(f"research queue drive: the lift stops below #{m.pr}: {e}")
                break
            if why and why.startswith("held"):
                break
            landing.append((m, why is None and not self._circuit_held(m)))
        lift = [m for m, ok in landing if ok]
        if all(chain[j] == m for j, m in enumerate(lift)):
            return []
        return [m.pr for m in lift]

    def _baselined(self, pl: dict) -> bool:
        """Whether `verified` ran the after-landing run as the review landing runs it, under `after_landing_no_cache`
        (`verified_as`: the steps it passed, every step when green, fewer when `main` was red then, `_red_baseline`), or ci
        recorded it green (`research queue baseline`, either flag)."""
        vs = pl.get("verified_as")
        return bool(vs) and vs.get("no_cache") in (None, self.after_landing_no_cache)

    def _unbaselined(self, drive: dict) -> str | None:
        """Why no review landing happens now, or None: `main` has no baseline (`_baselined`), so a failure after a landing couldn't be told from one `main` had before it.  The
        first run is then of `main` itself (`start_postland`), once any post-landing run still going has ended; ci records a green `main` by hand with `/data/infra/drive.sh queue baseline SHA --by ci`."""
        into = self.q.into
        main = _out(self.repo, "rev-parse", self.q.base)
        pl = drive.setdefault("postland", {"verified": main, "pending": [], "run": None})
        if self._baselined(pl):
            return None
        if run := pl.get("run"):
            return (f"nothing lands before the post-landing run {_ids(run)} of `{into}` {run['tip'][:12]} ends and an "
                    f"after-landing run of `{into}` itself has run (its baseline)")
        return f"nothing lands before an after-landing run of `{into}` {main[:12]} itself has run (its baseline)"

    def _bisect_step(self, drive: dict) -> None:
        """Bisect the landings a failed after-landing run implicates (`_postland_failed`, `postland["bisect"]`): check the commit
        of the middle one not yet judged with the failed steps alone (`--deferred STEPS`, the run's flags, `--base` where they
        last passed, below every landing it judges), each that passes clearing it and those below, each that fails blaming it
        and those above, until one fails and the one below it passes (or `verified` did): that PR is reverted
        (`_revert_landed`).  When even the newest passes, none of them broke it: nothing is reverted, `CI` is told.  A run
        fails a landing only when it failed one of those steps; one never started or stopped runs again, and one that ends
        otherwise (killed, lost, exit 0 without their records, a failure elsewhere) runs once more, then ends the bisection
        with nothing reverted."""
        if not (pl := drive.get("postland")) or not (b := pl.get("bisect")):
            return
        cands, tip, base = [Member(*x) for x in b["cands"]], b["tip"], b["base"]
        key, steps = f"postland:{tip}", ", ".join(b["steps"])
        head = (f"the after-landing run of `main` {tip[:12]} failed ({b['why']}) in {steps}; bisected "
                f"{' '.join(f'#{x.pr}' for x in cands)}, landed since {base[:12]}")
        if run := b.get("run"):
            c = cands[run["i"]].commit
            if (tail := self._resolve(run, f"bisecting `main` {tip[:12]}'s failed {steps}: #{cands[run['i']].pr}'s landing "
                                           f"{c[:12]}")) is not None:
                b["run"] = None
                self.refuse(drive, f"launch:bisect:{c}", f"train.sh {c[:12]} {self.on} --deferred {','.join(b['steps'])} "
                                                         f"registered no run: {tail}")
                return
            if "launch" in run:
                return
            states = self._states(run["ids"])
            lost = self._lost(run, states)
            if not lost and any(st in ACTIVE for st, *_ in states.values()):
                return
            b["run"] = None
            rcs = [rc for _, rc, *_ in states.values()]
            suite = self._suite_status(run["ids"], b["suite"]) if b.get("suite") and not lost else None
            if suite == "passed":
                b["good"] = run["i"]
            elif suite == "failed":
                b["bad"] = run["i"]
            elif b.get("suite") and not lost and all(rc in (NOT_STARTED, SIGTERM_RC) for rc in rcs):
                pass
            elif b.get("suite") and not b.get("unclear"):
                b["unclear"] = True
            elif b.get("suite"):
                self._bisected(drive, pl, b, None, f"its run {' '.join(run['ids'])} of {c[:12]} ended twice "
                                                   f"({'lost' if lost else ', '.join(map(str, rcs))}) neither passing nor "
                                                   f"failing {b['suite']}")
                return
            elif not lost and any(rc == 0 for rc in rcs) and not self._unrun(run["ids"], b["steps"]):
                b["good"] = run["i"]
            elif not lost and all(rc in (NOT_STARTED, SIGTERM_RC) for rc in rcs):
                pass
            elif not lost and not any(rc == 0 for rc in rcs) and set(self._failing(run)) & set(b["steps"]):
                b["bad"] = run["i"]
            elif not b.get("unclear"):
                b["unclear"] = True                     # killed, lost, or no failed step of its own read: once more
            else:
                pl["failed"] = tip
                del pl["bisect"]
                self.refuse(drive, key, f"{head}: its run {' '.join(run['ids'])} of {cands[run['i']].commit[:12]} ended twice "
                                        f"neither passing nor failing {steps} ({'lost' if lost else ', '.join(map(str, rcs))}). "
                                        "Reverted nothing; find the culprit by hand")
                return
        if b["bad"] - b["good"] == 1:
            one = cands[b["bad"]] if b["bad"] < len(cands) else None
            below = f"#{cands[b['good']].pr}'s landing" if b["good"] >= 0 else f"`main` {base[:12]}"
            self._bisected(drive, pl, b, one, f"it fails from #{one.pr}'s landing {one.commit[:12]} on and passes on {below}"
                           if one else f"each of {' '.join(f'#{x.pr}' for x in cands)}'s landings passes it")
            return
        mid = (b["good"] + b["bad"]) // 2
        c = cands[mid].commit
        extra = {"agreement": "lean-agreement" in b["steps"], "no_cache": b["no_cache"], "circuit_cache": b["no_cache"],
                 "no_regenerate": True}
        if self.dry_run or self.q.shadow:
            self.q.log(f"research queue drive: would launch train.sh {c[:12]} {self.on} --deferred {','.join(b['steps'])} to bisect")
            return
        rec, tail = self._started(self.launch_postland(self.repo, c, self.on, ",".join(b["steps"]), base=base, **extra))
        if rec is None:
            self.refuse(drive, f"launch:bisect:{c}", f"train.sh {c[:12]} {self.on} --deferred {','.join(b['steps'])} registered no "
                                                     f"run: {tail}")
            return
        b["run"] = {**rec, "i": mid, "at": self.now()}
        self.q.log(f"research queue drive: bisecting `main` {tip[:12]}'s failed {steps}: #{cands[mid].pr}'s landing "
                   f"{c[:12]}: {_ids(rec)}")
        self.q.save_state({**self.q.load_state(), "drive": drive})

    def _bisected(self, drive: dict, pl: dict, b: dict, one: Member | None, why: str) -> None:
        """End a bisection of red record ``b["key"]``: ``one`` is its culprit, rung now with its revert's clock started, or
        None, rung as unnamed; then the next red record with candidates is bisected."""
        del pl["bisect"]
        if (r := (pl.get("red") or {}).get(b["key"])) is not None:
            if one is not None:
                r.update(culprit=[one.pr, one.head, one.owner, one.commit], why=why, rung=self.now())
                self._ring(drive, r, b["key"], f"Culprit: #{one.pr}, by bisection: {why}")
            else:
                self._ring(drive, r, b["key"], f"Bisected {' '.join(f'#{c[0]}' for c in b['cands'])}: {why}, so none of them is "
                                               "named. Reverted nothing; find the culprit by hand", tag=":bisected")
        self._start_bisect(pl)

    def land_reviewed(self, drive: dict, dry_run: bool) -> tuple[Member | None, str]:
        """Land `next`'s longest prefix whose every member has a land verdict at its head and no hold (`queue.unreviewed`, by none
        of `_owners`), cut below a member whose Python doesn't compile (`_compiles`, run again on what is left below a cut): a
        fast-forward of `main` to its top, through `_land`.  (the member landed or None, what the tick did)"""
        q = self.q
        base, chain = self._chain()
        if not chain:
            return None, "nothing queued"
        main = _out(self.repo, "rev-parse", q.base)
        if main != base:
            return None, (f"`{q.into}` {main[:12]} moved off `{q.branch}`'s base {base[:12]}: it lands once a sync rebuilds "
                          f"`{q.branch}`")
        k, why = self._reviewed(chain)
        stop = f"#{chain[k].pr} at {chain[k].head[:8]}: {why}" if k < len(chain) else ""
        if not k:
            return None, f"nothing reviewed lands: {stop}"
        t = k - 1
        while t >= 0 and (cut := self._compiles(drive, chain, base, t)) < t:
            t = cut
        if t < 0:
            return None, f"nothing lands: #{chain[0].pr} at {chain[0].head[:8]} waits (see above)"
        m, members = chain[t], chain[: t + 1]
        prs = " ".join(f"#{x.pr}" for x in members)
        v = Verdict("passed", "", REVIEWED)
        if dry_run or q.shadow:
            q.log(f"research queue drive: would land {prs} ({m.commit[:12]}, {v.why})" + (" [shadow]" if q.shadow else ""))
            return m, f"would land #{m.pr}"
        got = self._land(drive, m, members, prs, v, main)
        return got, "" if got else f"{prs} reviewed but not landed (see above)" + (f"; {stop}" if stop else "")

    def _reviewed(self, chain: list[Member]) -> tuple[int, str | None]:
        """(how many members at the bottom of ``chain`` land on their verdicts, why the next one doesn't): each has its ready mark
        and a land verdict at its head and no hold (`queue.unreviewed`, by none of `_owners`), and a red circuit-check doesn't
        hold it (`_circuit_held`)."""
        for k, m in enumerate(chain):
            if not self.q.ready(m.pr, m.head):           # an eject writes `ready = false`; a re-mark at that head is newer
                return k, "its ready mark is withdrawn: it leaves `next` at the next rebuild"
            if (why := unreviewed(self.reviews(m.pr, m.head), self._owners(m))) is not None:
                return k, why
            if why := self._circuit_held(m):
                return k, why
        return len(chain), None

    def _circuit_held(self, m: Member) -> str | None:
        """Why ``m`` waits while circuit-check is red on `checked` (`CIRCUIT_RED`): its own change touches circuit-check's inputs
        (`queue.forces_cold`, `_input_rule`'s rule for it, which takes all of `verity/ml/circuits/` and
        `tools/verity/circuit_check/` but their tests and docs), and its body doesn't say `fixes step circuit-check` (the fix
        of a red no culprit was named for has to land).  None otherwise."""
        if CIRCUIT_RED not in ((self._drive or {}).get("postland") or {}).get("red", {}) or not forces_cold(self._own(m)):
            return None
        if (p := self._open().get(m.pr)) and re.search(r"\bfixes\s+step\s+`?circuit-check(?![\w/.-])", p.body or "", re.I):
            return None
        return f"circuit-check is red on `{self.q.into}`, and its change touches circuit-check's inputs: it lands once that passes"

    def _owners(self, m: Member) -> set[str | None]:
        """Whose verdict on ``m`` doesn't count (`queue.review_owners`): its merge's `Owner`, that owner's handle (`_owner`), and
        who wrote its ready mark."""
        return {m.owner, self._owner(m), self.ready_by(m.pr, m.head)}

    def _compiles(self, drive: dict, chain: list[Member], base: str, t: int) -> int:
        """The highest index at or below ``t`` whose landing compiles: every .py file ``base``..``chain[t]`` leaves, compiled as
        `python -m py_compile` does.  The member that last changed a file that doesn't is ejected (`_eject`), and what's below
        it lands."""
        top = chain[t].commit
        bad = []
        for f in sorted(changed(self.repo, base, top)):
            if not f.endswith(".py") or (r := _git(self.repo, "show", f"{top}:{f}", check=False)).returncode:
                continue
            try:
                compile(r.stdout, f, "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as e:
                bad.append((f, f"{type(e).__name__}: {e}"))
        if not bad:
            return t
        by: dict[int, list[str]] = {}
        for f, err in bad:
            i = max(i for i in range(t + 1) if f in changed(self.repo, f"{chain[i].commit}^1", chain[i].commit))
            by.setdefault(i, []).append(f"{f}: {err}")
        u = min(by)
        prs = " ".join(f"#{x.pr}" for x in chain[: t + 1])
        self._eject(drive, f"compile:{chain[u].commit}", f"`{self.q.branch}` {top[:12]} ({prs}) doesn't compile, the gate before a "
                    "review landing", chain[t], prs, Verdict("failed", "py_compile"), [chain[i] for i in sorted(by)],
                    "; ".join(f"#{chain[i].pr}: {', '.join(errs)}" for i, errs in sorted(by.items())))
        return u - 1

    def _after_landing(self, drive: dict, tip: str, base: str) -> None:
        """A review landing at ``tip`` on ``base`` awaits the after-landing run: every check step on `main` (`start_postland`)."""
        pl = drive.setdefault("postland", {"verified": base, "pending": [], "run": None})
        pl["pending"].append(tip)
        pl.setdefault("landed_at", {})[tip] = self.now()
        self.q.save_state({**self.q.load_state(), "drive": drive})

    def tick(self, dry_run: bool = False) -> str:
        self._after: list[Callable[[], None]] | None = []    # the landings' after-steps, run once the tick's launches are out
        try:
            return self._tick(dry_run)
        finally:
            self._run_after()

    def _tick(self, dry_run: bool) -> str:
        q = self.q
        self.dry_run, self._seen, self._prs, self._ev = dry_run, {}, None, {}
        self._reg, self._own_files = None, {}
        q._clone()
        q.fetch()
        if q.store is not None and q.store.remote is not None:
            try:
                q.store.refresh(attempts_only=True)
            except Exception as e:  # noqa: BLE001  (the local catalog still answers)
                q.log(f"research queue drive: refreshing attempts failed ({type(e).__name__}: {e}); using the local catalog")
        state = q.load_state()
        drive = state.setdefault("drive", {})
        drive.setdefault("told", [])
        for gone in TRAIN_STATE:
            drive.pop(gone, None)
        self._drive = drive
        self.follow_postland(drive)
        return self._review(state, drive, dry_run)

    def _sweep(self, drive: dict, dry_run: bool) -> None:
        q = self.q
        if not dry_run and not q.shadow and self.now() - drive.get("branches_swept", 0) > BRANCH_SWEEP_S:
            from .merge import SWEEP_LIMIT, sweep_branches

            with authed(q.auth):
                deleted, archived = sweep_branches(self.repo, "origin", log=q.log, keep=drive.get("stale_closed", {}))
            more = SWEEP_LIMIT in (deleted, archived)
            if not more:
                drive["branches_swept"] = self.now()
            q.log(f"research queue drive: branch sweep: {deleted} deleted (on main), {archived} archived (unmerged, idle)"
                  + (f"; at its limit of {SWEEP_LIMIT}, so the next tick sweeps again" if more else ""))

    # -- what only a PR's owner acts on: a stale PR, a stacked PR whose base landed

    def _open(self) -> dict[int, PR]:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self._prs

    def tell(self, drive: dict, keys: list[str], text: str, handle: str | None) -> None:
        """Tell ``handle``, a PR's owning lane (`CI` when no handle owns it), unless every one of ``keys`` was told; unlike
        `refuse`, not `CI` beside it.  With --shadow or --dry-run, only log it."""
        if keys and all(k in drive["told"] for k in keys):
            return
        to = (f"@{handle}",) if handle else (CI,)
        if self.q.shadow or self.dry_run:
            self.q.log(f"research queue drive: would tell {to[0]}: {text} [{'shadow' if self.q.shadow else 'dry run'}]")
            return
        if self.post(f"{to[0]} queue: {text}", to):
            drive["told"].extend(k for k in keys if k not in drive["told"])
            self.q.log(f"research queue drive: told {to[0]}: {text}")
        else:
            self.q.log(f"research queue drive: could not post to the merge thread (the next tick retries): {to[0]}: {text}")

    def _stale(self, drive: dict, dry_run: bool) -> None:
        """The stale rule (`research.stale`): ring each stale PR's owner once a UTC day (`idle:N@DAY`), one post per owner, and
        close each PR `stale_close_days` idle with `stale.close_comment`, but not one on `next` or a draft that says "keep open"
        (`stale.keep_open`); a closed PR's branch is kept (`stale_closed`, which `_sweep` never archives).  Ring keys of other
        days, and the stacked-PR keys (`_stacked`) of PRs no longer open, leave `told`.  At most once in `STALE_EVERY_S`."""
        q, now = self.q, self.now()
        if not self.stale_days or now - drive.get("stale_judged", 0) < STALE_EVERY_S:
            return
        from .stale import DAY_S, close_comment, keep_open, stale_prs

        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        try:
            prs = self._open()
            idle = stale_prs(q, prs.values(), now, self.stale_days, lambda n, h: [lab.ts for lab in self.reviews(n, h)],
                             keep=self.stale_keep)
        except QueueError as e:
            q.log(f"research queue drive: stale PRs not judged this tick: {e}")
            return
        drive["told"] = [k for k in drive["told"] if not (k.startswith("idle:") and not k.endswith(f"@{day}"))
                         and not (k.startswith("stacked:") and int(k[len("stacked:"):].partition("<")[0]) not in prs)]
        if not (dry_run or q.shadow):
            drive["stale_judged"] = now
        on_next = {m.pr for m in q.chain()}
        ring: dict[str | None, list[str]] = {}
        keys: dict[str | None, list[str]] = {}
        closed: dict[str | None, list[str]] = {}
        for n, since in sorted(idle.items()):
            p, days = prs[n], int((now - since) // DAY_S)
            if n in on_next or keep_open(p):
                continue
            who = self._handle(n)
            if self.stale_close_days and now - since >= self.stale_close_days * DAY_S:
                if dry_run or q.shadow:
                    q.log(f"research queue drive: would close #{n}: {days} days unmarked and idle")
                    continue
                try:
                    q.github.close(n, close_comment(self.stale_close_days))
                except QueueError as e:
                    q.log(f"research queue drive: closing #{n} ({days} days unmarked and idle) failed; the next tick tries again: {e}")
                    continue
                if p.branch:
                    drive.setdefault("stale_closed", {})[p.branch] = n
                q.log(f"research queue drive: closed #{n}: {days} days unmarked and idle")
                closed.setdefault(who, []).append(f"#{n}")
            elif f"idle:{n}@{day}" not in drive["told"]:
                ring.setdefault(who, []).append(f"#{n} ({days} d)")
                keys.setdefault(who, []).append(f"idle:{n}@{day}")
        for who in dict.fromkeys([*ring, *closed]):
            said = []
            if ring.get(who):
                said.append(f"stale, not marked ready at their heads and idle {self.stale_days}+ days: {', '.join(ring[who])}. Push, "
                            "mark ready or close each" + (f"; drive closes one {self.stale_close_days} days idle, its branch kept"
                                                          if self.stale_close_days else ""))
            if closed.get(who):
                said.append(f"closed after {self.stale_close_days} days unmarked and idle, branches kept (reopen one to go on): "
                            f"{', '.join(closed[who])}")
            self.tell(drive, keys.get(who, []), "; ".join(said), who)

    def _stacked(self, drive: dict, members: list[Member], tip: str) -> None:
        """After a landing at ``tip``, tell the owner of each open PR whose head contains a landed member's head (a PR stacked
        on it) that its base landed, once per (PR, base) (`stacked:P<Q`): not one marked ready at its head, which lands from
        `next` as it is, and whose merge of `main` would need a new mark."""
        from .stale import pull_refs

        q = self.q
        landed = {x.pr for x in members}
        try:
            rest = {n: p for n, p in self._open().items() if n not in landed}
            refs = pull_refs(q, rest.values())
        except QueueError as e:
            q.log(f"research queue drive: PRs stacked on {', '.join(f'#{n}' for n in sorted(landed))} not found: {e}")
            return
        bases: dict[int, list[int]] = {}
        for x in members:
            out = _git(self.repo, "for-each-ref", "--contains", x.head, "--format=%(refname:lstrip=3)", "refs/queue/pr/").stdout
            for n in (int(s) for s in out.split() if s.isdigit()):
                if n in rest and refs.get(n) == rest[n].head and f"stacked:{n}<{x.pr}" not in drive["told"]:
                    bases.setdefault(n, []).append(x.pr)
        for n, below in sorted(bases.items()):
            try:
                if q.ready(n, rest[n].head):
                    continue
            except QueueError as e:
                q.log(f"research queue drive: #{n}'s ready mark can't be read, so it isn't told its base landed now: {e}")
                continue
            names = ", ".join(f"#{b}" for b in below)
            self.tell(drive, [f"stacked:{n}<{b}" for b in below],
                      f"#{n}'s base{'s' if len(below) > 1 else ''} {names} landed in main at {tip[:12]}; merge main into #{n}",
                      self._handle(n))

    def _run_after(self) -> None:
        pending, self._after = self._after, None
        for step in pending or ():
            step()

