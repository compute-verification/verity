"""What the verified queue's loop (`research.vqueue_loop`) and the run beside each candidate's check (`research.vqueue_slim`)
need beyond graph2: who hears what (`CI`, `INFRA`, the merge thread: `merge_thread_post`), the run beside's launch
(`train_launch`: `train.sh SHA MACHINES --deferred STEPS`, detached by `detach`, its run ids read back by `launched`) and its
records read back (`run_states`, `evidence`: steps.jsonl, the suites' reports, what a failure names: `implicated`,
`step_paths`, `killed`), the steps `main` waits for (`gating`), and `Landing`: policy.toml's allowances of a failed step or
graph/v2 task, the owners of a path and of a job, and what follows a candidate's members reaching `main` (`Landing._on_main`).

    Landing(q, on=MACHINES)._task_allowed(sha, task, result, row)  -> policy.Coverage of a failed graph/v2 task
    Landing(q, on=MACHINES)._task_units(sha, task, result)         -> {red record: its suite's report row}
    Landing(q, on=MACHINES)._path_owner(path), _step_owner(job, sha) -> the handle that owns it, or None
    Landing(q, on=MACHINES)._on_main(state, members, base, tip, "") -> stacked PRs told, landed PRs caught up, branches swept
    Landing(q, on=MACHINES).beside_args(sha, base)                  -> (the run beside's keywords, its skipped steps, its flags)

One `Landing` outlives the loop's ticks: what it reads once (`_states`, `_evidence`, `_policy`, the registry) it keeps, so
`vqueue_slim.judge` reads the runs' states afresh each call."""
from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import policy
from .queue import CHECK_STEPS, PR, Member, Queue, QueueError, _git

#: who hears what the loop can't settle, and who ejects
CI = "@ci"

#: who hears that a machine, a store or graph2 failed
INFRA = "@infra"

#: a pytest location (`tests/test_x.py:503: NameError`) or a traceback frame (`File "/.../x.py", line 12`) in a suite's log
TRACE_RE = re.compile(r'^\s*([\w./+-]+\.py):\d+:|File "([^"]+\.py)", line \d+', re.M)

#: a repository path a pytest failure report names (`E   tools/x/status.md`: a lint's offender is in its assertion, not its
#: traceback); relative, with a directory
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
#: imports outside its layer (neither names a file the other patterns match)
LEAN_ORPHAN_RE = re.compile(r"^\s+orphan: (\S+?\.lean) is not imported")

LEAN_LAYERS_RE = re.compile(r"^\s+layers: ([A-Za-z][\w.]*) imports ")

#: a run's or a step's exit that a kill gave, not a failure of what it checked: SIGHUP, SIGKILL, SIGTERM as 128+N, or a signal
KILLED_RC = (129, 137, 143)

#: what the Lean check says of a process of its own that was killed (`tools/verity/lean/check.py`: the replay, `--fresh`)
KILLED_SAYS = "not a kernel verdict"

#: circuit-check's report of a failure: `FAIL <check> <target>`, and `circuit-check/killed` for a target whose process died
CIRCUIT_FAIL_RE = re.compile(r"^FAIL (\S+)", re.M)

CIRCUIT_KILLED = "circuit-check/killed"

#: states `research status` reports for a run that may still publish an attempt
ACTIVE = ("submitted", "waiting", "running", "unknown")

#: check.py's exit for a twin that never took a slot
NOT_STARTED = 75

#: the check step that runs the vLLM integration's ratchet gate on the merges above the check's base
RATCHET_STEP = "ratchet-gate"

#: the check's jobs (`--deferred` takes jobs; the preflights are none), those off the merge gate (policy.toml's `off_gate`)
#: among them
JOBS = tuple(s for s in CHECK_STEPS if not s.startswith("preflight-"))

#: the jobs a check may skip by name: lean-agreement when nothing it checks changed, lean-changed under --no-cache
SKIPPABLE = ("lean-agreement", "lean-changed")

#: the steps the run beside each candidate's check runs (`research.vqueue_slim`), which no graph/v2 task checks: lean-agreement,
#: unless nothing it checks changed, and ratchet-gate, off the merge gate (policy.toml's `off_gate`); graph2's tasks of them count
#: for nothing
SLIM_STEPS = ("lean-agreement", RATCHET_STEP)

#: what the run beside a candidate's check judges of its records (`vqueue_slim.judge`): `SLIM_STEPS`, and the preflights no
#: graph/v2 task runs; the steps they need (lean-build) are graph2's tasks' to judge
SLIM_JUDGED = (*SLIM_STEPS, *(s for s in CHECK_STEPS if s.startswith("preflight-")))

#: how long a run may show no readable state (`unknown`) before it is taken as lost (`Landing._lost`)
LOST_AFTER_S = 3 * 3600

#: circuit-check's red record, which an allowance covers by the failures its log names (`Landing._allowed`)
CIRCUIT_RED = "step circuit-check"

#: the ci.toml job a graph/v2 task's kind checks (`verity graph`'s families, tools/verity/graph/tasks.toml), by which
#: policy.toml takes it off the merge gate (`off_gate`) and its red is named `step JOB` (a test-file task's `suite NAME`); a kind
#: not listed is its own job
JOB_OF_KIND = {"test-file": "pytest", "lean-module": "lean-audit", "lean-lock": "lean-audit", "circuit-binding": "circuit-check",
               "circuit-coverage": "circuit-check", "circuit-catalog": "circuit-check"}

#: the graph/v2 kind that runs check's lean-suites step on one of its suites, or a part of one (`verity_graph.runners.lean_suites`):
#: its fail names each failed suite (`detail.failed`, suites.py's rows), which the loop names, allows and blames as that
#: suite's own red, a red unit any of a suite's parts fails
SUITES_KIND = "lean-suites"

#: the most new failures a circuit-binding result names (`verity_graph.runners.circuit_binding.NAMED`): a list that long may
#: leave some out, so no allowance covers it
CIRCUIT_NAMED = 100

#: a run stopped by SIGTERM (128 + 15): a cancel, or a twin whose sibling won the slot
SIGTERM_RC = 143

RUN_RE = re.compile(r"^RUN ((?:r20\d{6}-\d{6}-[0-9a-f]{4} ?)+)$", re.M)


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
    """os.environ with the queue's credentials (`auth_env`) and git never prompting, for the steps after an advance
    (`merge.caught_up`, `_sweep_after`), whose git children read os.environ (a git that prompts, as `caught_up`'s `ls-remote` in the
    queue's clone would, waits on a password at the tty and holds the loop)."""
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


def train_launch(repo: Path, commit: str, machine: str, agreement: bool = False, auth: tuple[str, ...] = (),
                 deferred: str | None = None, no_regenerate: bool = False, detached: bool = False,
                 base: str | None = None, rank: int | None = None) -> tuple[list[str], str] | dict:
    """`train.sh COMMIT MACHINE [--agreement] [--deferred STEPS] [--no-regenerate] [--base BASE]` from the queue's clone:
    (the run ids it printed, its output's tail); ``detached``, started in the background (`detach`) from a copy of train.sh,
    which reads nothing by its own path (``$0`` only for its usage) and finds the tree from its working directory, and returned
    at once as the record `launched` reads.  Git never prompts (the queue's clone has no credential helper, so train.sh's fetch
    would wait on a password at the tty).  ``rank`` is its check's `slot.py --rank` (CHECK_SLOT_RANK), none for an unranked
    one."""
    argv = ["bash", "tools/verity/check/train.sh", commit, machine, *(["--agreement"] if agreement else []),
            *(["--deferred", deferred] if deferred else []), *(["--no-regenerate"] if no_regenerate else []),
            *(["--base", base] if base else [])]
    env = {**os.environ, **auth_env(auth), "GIT_TERMINAL_PROMPT": "0", "TRAIN_AS": "ci",
           "CHECK_SLOT_RANK": "" if rank is None else str(rank)}
    if detached:
        return detach(repo, argv, env, script=argv[1])
    r = subprocess.run(argv, cwd=repo, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    m = RUN_RE.search(r.stdout)
    return (m[1].split() if m else []), (r.stdout + r.stderr).strip()[-600:]


#: the check slot rank (`slot.py --rank`) of the run beside each candidate's check: a ranked priority waiter takes the next
#: free check slot ahead of the unranked ones, so no candidate waits on a slot for it
SLOT_RANK = 0

#: where detached launches (`detach`) keep their output and exit, under the loop's TMPDIR
LAUNCHES = "vqueue-launches"

#: how long a detached launch may print no run ids before it is taken as having registered none: train.sh stops its launcher
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

#: a check's trace/v1 spans (`tools/verity/check/spans.py`), whose root span opens as the check starts its steps
SPANS = "spans.jsonl"

#: a check's live record (`tools/verity/check/live.py`): each step it planned, pending until it starts and running until it ends
LIVE = "check-live.json"

#: the steps a check runs first, one after another, before its groups start (check.py's `preflight`)
PREFLIGHT = tuple(s for s in CHECK_STEPS if s.startswith("preflight-"))


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


def gating(repo: Path, commit: str) -> list[str]:
    """The check's jobs `main` waits for at ``commit``: every one but those a check skips by name (`SKIPPABLE`) and the jobs
    ``commit``'s own policy takes off the merge gate (policy.toml's `off_gate`: circuit-check, `merge.declared_jobs`); every such
    job when that ci.toml or policy can't be read."""
    from .merge import declared_jobs

    jobs = declared_jobs(repo, commit)
    return [s for s in JOBS if s not in SKIPPABLE and (isinstance(jobs, str) or s in jobs)]


def run_states(repo: Path, ids: list[str]) -> dict[str, tuple]:
    """Each run's (state, rc, ended_utc, what has failed so far while it runs: `failing`, the machine it runs on) from `research
    status RUN --json`, which fetches `PROGRESS` and a check's live record (`LIVE`) with its records, so those of a run that ended
    are its last (a check waiting in its machine's slot line reads `waiting`); a run it can't read is ("unknown", None, None, None)."""
    also = [x for n in (*PROGRESS, LIVE) for x in ("--also", n)]

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


#: how many runs are read at once: each `research status` is an ssh round trip to the run's node (reading a few
#: dozen one at a time takes a tick half an hour)
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
    """The objects the JSON files ``names`` of ``run_dir`` hold: a list's, or the file's own; none of one missing or unreadable."""
    out = []
    for name in names:
        try:
            doc = json.loads((run_dir / name).read_text())
        except (OSError, ValueError):
            continue
        out += [x for x in doc if isinstance(x, dict)] if isinstance(doc, list) else [doc] if isinstance(doc, dict) else []
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
    """The local run dirs of ``ids`` with their steps, live records, spans, suite reports and failed suites' logs fetched
    (`research status --also`)."""
    def one(rid: str) -> Path | None:
        if (d := _fetch(repo, rid, [*PROGRESS, SPANS, LIVE, "suites.json", "suites-lean.json"])) is None:
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


def suite_pyproject(repo: Path, commit: str, suite: str) -> dict:
    """Suite ``suite``'s (its directory's) pyproject.toml at ``commit``; {} when it has none or it isn't TOML."""
    import tomllib

    own = "pyproject.toml" if suite.strip("/") in ("", ".") else f"{suite.strip('/')}/pyproject.toml"
    r = _git(repo, "show", f"{commit}:{own}", check=False)
    try:
        return tomllib.loads(r.stdout) if not r.returncode else {}
    except tomllib.TOMLDecodeError:
        return {}


def suite_name(repo: Path, commit: str, suite: str) -> str:
    """The name `suites.py` gives the suite in directory ``suite`` at ``commit`` (its `discover`): `repository` for the root,
    else its pyproject.toml's project name, normalized, else its path."""
    if suite.strip("/") in ("", "."):
        return "repository"
    name = suite_pyproject(repo, commit, suite).get("project", {}).get("name")
    return re.sub(r"[-_.]+", "-", name).lower() if isinstance(name, str) else suite


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


#: the step whose suites each report of a check holds, which a suite's red record keeps as its `step`
SUITE_REPORTS = {"suites.failed.json": "pytest", "suites-lean.failed.json": "lean-suites", "suites.json": "pytest",
                 "suites-lean.json": "lean-suites"}


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


def touches(files: set[str], paths: set[str]) -> set[str]:
    """The ``files`` (repository paths) among ``paths``, an absolute path matching the file it ends in when that file is under a
    directory (a bare `conftest.py` would match every conftest an installed package has)."""
    return {f for f in files for p in paths if p == f or ("/" in f and p.startswith("/") and p.endswith("/" + f))}


class Landing:
    """The loop's side of a landing on queue ``q`` (the module's docstring).  ``on`` is drive.toml's machines at `main`, which the
    run beside each check runs on ("" for one that launches nothing); ``launch`` launches it (`train_launch`, detached, its
    check at `SLOT_RANK`); ``states``, ``evidence`` and ``cancel`` read and stop runs (`run_states`, `evidence`, `cancel_runs`);
    ``post`` posts to the merge thread (`merge_thread_post`); ``now`` is the clock."""

    def __init__(self, q: Queue, *, on: str, launch: Callable[..., tuple[list[str], str] | dict] | None = None,
                 states: Callable[[Path, list[str]], dict] = run_states,
                 post: Callable[[str, tuple[str, ...]], bool] = merge_thread_post, now: Callable[[], float] = time.time,
                 cancel: Callable[[Path, list[str], str], dict[str, str] | None] = cancel_runs,
                 evidence: Callable[[Path, list[str]], list[Path]] = evidence):
        self.q, self.on = q, on
        #: the run beside a check (`vqueue_slim.start`): `train.sh COMMIT MACHINES --deferred STEPS`, detached (`_started`), its
        #: check at `SLOT_RANK`
        self.launch: Callable[..., tuple[list[str], str] | dict] = launch or (
            lambda repo, commit, machine, steps, **kw: train_launch(repo, commit, machine, auth=q.auth, deferred=steps,
                                                                     detached=True, rank=SLOT_RANK, **kw))
        #: what a detached launch has printed so far (`launched`, `_resolve`)
        self.launched: Callable[[dict], tuple[list[str] | None, str]] = launched
        self.states, self.post, self.now, self.cancel, self.evidence = states, post, now, cancel, evidence
        self.repo, self.dry_run, self._seen, self._prs, self._ev = q.workdir, False, {}, None, {}

    def _states(self, ids: list[str]) -> dict:
        """`states` of ``ids``, each run read once a tick: one `research status` is a few seconds over ssh, and a tick
        reads many."""
        if missing := [rid for rid in ids if rid not in self._seen]:
            self._seen.update(self.states(self.repo, missing))
        return {rid: self._seen[rid] for rid in ids}

    @staticmethod
    def _started(got: tuple[list[str], str] | dict) -> tuple[dict | None, str]:
        """A launch record's fields from what `launch` returned: {"ids"} of a launch that printed them, {"ids": [], "launch"} of
        one still running detached (`detach`), which `_resolve` reads on later ticks and which counts as running until then;
        None, with train.sh's output's tail, of one that registered no run."""
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
        self.q.log(f"research queue vqueue: {what}: {' '.join(ids)}")
        return None

    def _evidence(self, ids: list[str]) -> list[Path]:
        """`evidence` of runs ``ids``, fetched once."""
        if (key := " ".join(ids)) not in self._ev:
            self._ev[key] = self.evidence(self.repo, ids)
        return self._ev[key]

    def _ancestor(self, a: str, b: str) -> bool:
        return not _git(self.repo, "merge-base", "--is-ancestor", a, b, check=False).returncode

    @staticmethod
    def _job(task: dict) -> str:
        """The ci.toml job graph/v2 ``task`` checks (`JOB_OF_KIND`)."""
        return JOB_OF_KIND.get(task["kind"], task["kind"])

    def _suite_name(self, tip: str, suite: str) -> str:
        """`suite_name` of the suite in directory ``suite`` at ``tip``, read once per tick."""
        cache = self.__dict__.setdefault("_suite_names", {})
        if (tip, suite) not in cache:
            cache[tip, suite] = suite_name(self.repo, tip, suite)
        return cache[tip, suite]

    def _task_unit(self, tip: str, task: dict) -> str:
        """The red record a failed graph/v2 ``task`` of ``tip`` is named in, the one a run's failure would be: a test-file's
        `suite NAME`, any other's `step JOB` (`_job`; circuit-check's is `CIRCUIT_RED`)."""
        if task["kind"] == "test-file":
            return f"suite {self._suite_name(tip, (task.get('vars') or {}).get('suite') or '.')}"
        return f"step {self._job(task)}"

    def _task_units(self, tip: str, task: dict, result: dict) -> dict[str, dict | None]:
        """The red records failed graph/v2 ``task`` names, each with its suite's report row: a lean-suites task's, each failed
        suite its `detail.failed` names, `suite NAME` with its row, as a run's lean-suites failures are named; else, and when it
        names none (a timeout of the whole task), `_task_unit`'s alone."""
        rows = (result.get("detail") or {}).get("failed") if task["kind"] == SUITES_KIND else None
        named = {f"suite {r['name']}": r for r in rows or () if isinstance(r, dict) and isinstance(r.get("name"), str)}
        return named or {self._task_unit(tip, task): None}

    def _task_allowed(self, tip: str, task: dict, result: dict, row: dict | None = None) -> policy.Coverage:
        """Whether an allowance of ``tip``'s policy covers failed graph/v2 ``task`` by its ``result``, with the expired ones that
        would have: a failed suite of a lean-suites task's by its report ``row`` (`Policy.suite`, as a run's); a test-file's as
        its suite's (`Policy.suite`, `suite NAME` or `suite NAME::TEST` for each failed test, the latter only when pytest exited
        1 with failed tests to name); a circuit-check task's by the new failures it names (`Policy.circuit`), fewer than
        `CIRCUIT_NAMED`; no other kind's."""
        pol, now, d = self._policy(tip), self.now(), result.get("detail") or {}
        if row is not None:
            return pol.suite(str(row["name"]), row, now)
        if task["kind"] == "test-file":
            tests = [str(x) for x in d.get("failed_tests") or ()]
            row = {"failed_tests": tests, "beyond_tests": not (d.get("exit") == 1 and bool(tests))}
            return pol.suite(self._task_unit(tip, task).split(" ", 1)[1], row, now)
        if self._job(task) == "circuit-check" and len(names := [str(x) for x in d.get("failures") or ()]) < CIRCUIT_NAMED:
            return pol.circuit("".join(f"FAIL {n}\n" for n in names), now)
        return policy.Coverage(False)

    def _shards(self, run: dict) -> list[list[Path]]:
        """The run dirs of each shard of ``run`` (``groups``, its twins', by run id; an unsharded run is one), fetched once a
        tick."""
        dirs = self._evidence(run["ids"])
        return [[d for d in dirs if d.name in g] for g in run["groups"]] if run.get("groups") else [dirs]

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

    def _unrun(self, ids: list[str], steps: list[str], tip: str | None = None) -> list[str]:
        """The ``steps`` no run among ``ids`` records as passed in its steps.jsonl, nor (with ``tip``, the commit it checked)
        as failed only on allowed failures (`_allowed_steps`): a run's exit 0 says only that nothing failed (a step
        `skipped:`, or one `--deferred` maps to no part, passes it too)."""
        got, dirs = set(), self.evidence(self.repo, ids)
        for d in dirs:
            with contextlib.suppress(OSError):
                for line in (d / PROGRESS[0]).read_text().splitlines():
                    with contextlib.suppress(ValueError):
                        if (rec := json.loads(line)).get("status") == "passed":
                            got.add(rec.get("step"))
        if tip is not None:
            got |= self._allowed_steps(tip, dirs)
        return [s for s in steps if s not in got]

    def _policy(self, commit: str) -> policy.Policy:
        """``commit``'s policy (policy.toml), read once; one that can't be read is logged and taken as none (every job gates, no
        failure is allowed)."""
        cache = self.__dict__.setdefault("_policy_of", {})
        if commit not in cache:
            got = policy.at(self.repo, commit)
            if isinstance(got, str):
                self.q.log(f"research queue vqueue: {got}: no failure is allowed at {commit[:12]}")
                got = policy.Policy()
            cache[commit] = got
        return cache[commit]

    def _allowed(self, tip: str, key: str, d: Path, rec: dict) -> policy.Coverage:
        """Whether an allowance of ``tip``'s policy covers failed ``key`` (`suite NAME`, or a step: circuit-check's by its
        log's failures), with the expired ones that would have."""
        pol, now = self._policy(tip), self.now()
        if key.startswith("suite "):
            return pol.suite(key.split(" ", 1)[1], rec, now)
        if key == CIRCUIT_RED:
            return pol.circuit(_step_log(d, rec), now)
        return policy.Coverage(False)

    def _allowed_steps(self, tip: str, dirs: list[Path]) -> set[str]:
        """The steps that failed in the run whose records are ``dirs`` only on failures ``tip``'s allowances cover: pytest and
        lean-suites when each suite of theirs that failed is covered, circuit-check when each failure its log names is; none
        that failed uncovered in any of them."""
        ok: set[str] = set()
        bad: set[str] = set()
        for d in dirs:
            rows = suite_rows(d)
            for st in step_records(d):
                step = st.get("step") or ""
                if st.get("status") != "failed" or not_run(st):
                    continue
                if step in ("pytest", "lean-suites"):
                    failed = [(n, s) for n, s in rows.items() if s["status"] == "failed" and s["step"] == step]
                    covered = bool(failed) and all(self._allowed(tip, f"suite {n}", d, s).covered for n, s in failed)
                else:
                    covered = self._allowed(tip, f"step {step}", d, st).covered
                (ok if covered else bad).add(step)
        return ok - bad

    def _only_allowed(self, run: dict) -> bool:
        """Whether ``run`` failed only on failures its tip's allowances cover: some step failed, and each failed step is
        `_allowed_steps`'."""
        dirs = [d for ds in self._shards(run) for d in ds]
        failed = {st.get("step") for d in dirs for st in step_records(d) if st.get("status") == "failed" and not not_run(st)}
        return bool(failed) and failed <= self._allowed_steps(run["tip"], dirs)

    def _lost(self, run: dict, states: dict) -> bool:
        """Whether every twin of ``run`` that hasn't ended has had no readable state (`unknown`) for `LOST_AFTER_S` since its
        launch: it is taken as lost, as a killed run is, so a store or node that never answers doesn't hold the loop forever."""
        live = [st for st, *_ in states.values() if st in ACTIVE]
        return bool(live) and all(st == "unknown" for st in live) and self.now() - run.get("at", self.now()) > LOST_AFTER_S

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

    def _on_main(self, state: dict, members: list[Member], since: str, tip: str, attempt: str) -> None:
        """What follows ``members`` reaching `main` at ``tip`` from ``since``: the owner of each open PR stacked on one hears that
        its base landed (`_stacked`, each key it tells added to ``state["told"]``), each landed PR is caught up (`merge.caught_up`:
        closed as landed, its issue moved, its after-land command given the check ``attempt``) and the branches landed since are
        swept (`merge._sweep_after`), but those ``state["stale_closed"]`` keeps."""
        from .merge import _sweep_after, caught_up

        q = self.q
        self._stacked(state, members, tip)
        with authed(q.auth):
            caught_up(self.repo, "origin", "check", since=since, log=q.log, runs={x.commit: attempt for x in members} | {tip: attempt})
            _sweep_after(self.repo, "main", "origin", keep=state.get("stale_closed", {}))

    def _jobs(self, commit: str) -> list[dict]:
        """ci.toml's job tables at ``commit`` (`merge.declared`), read once a tick; none when it can't be read."""
        from .merge import declared

        cache = self.__dict__.setdefault("_jobs_at", {})
        if commit not in cache:
            cache[commit] = declared(self.repo, commit) or []
        return cache[commit]

    def _step_owner(self, step: str, commit: str) -> str | None:
        """The owner ci.toml at ``commit`` gives job ``step``, None for a step it doesn't declare (a preflight)."""
        return next((t["owner"] for t in self._jobs(commit) if t["name"] == step and isinstance(t.get("owner"), str)), None)

    def _handle(self, pr: int) -> str | None:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self.q.handle(self._prs[pr]) if pr in self._prs else None

    def _open(self) -> dict[int, PR]:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self._prs

    def tell(self, state: dict, keys: list[str], text: str, handle: str | None, *also: str | None) -> None:
        """Tell ``handle``, a PR's owning lane (`CI` when no handle owns it), and ``also`` in the same post, unless every one of
        ``keys`` was told (``state["told"]``, which it extends).  With --shadow or --dry-run, only log it."""
        if keys and all(k in state["told"] for k in keys):
            return
        to = tuple(dict.fromkeys((f"@{handle}" if handle else CI, *(f"@{h}" for h in also if h))))
        if self.q.shadow or self.dry_run:
            self.q.log(f"research queue vqueue: would tell {' '.join(to)}: {text} [{'shadow' if self.q.shadow else 'dry run'}]")
            return
        if self.post(f"{' '.join(to)} queue: {text}", to):
            state["told"].extend(k for k in keys if k not in state["told"])
            self.q.log(f"research queue vqueue: told {' '.join(to)}: {text}")
        else:
            self.q.log(f"research queue vqueue: could not post to the merge thread (the next tick retries): {' '.join(to)}: {text}")

    def _stacked(self, state: dict, members: list[Member], tip: str) -> None:
        """After an advance to ``tip``, tell the owner of each open PR based on `main` or on a landed member's branch whose head
        contains that member's head (a PR stacked on it) that its base landed, once per (PR, base) (`stacked:P<Q`): not one
        marked ready at its head, which the loop builds as it is.  A PR based
        on another branch (a long-lived one, as the remote demo's) takes `main` through that branch, and isn't told."""
        from .stale import pull_refs

        q = self.q
        landed = {x.pr for x in members}
        try:
            opened = self._open()
            rest = {n: p for n, p in opened.items() if n not in landed}
            refs = pull_refs(q, rest.values())
        except QueueError as e:
            q.log(f"research queue vqueue: PRs stacked on {', '.join(f'#{n}' for n in sorted(landed))} not found: {e}")
            return
        bases: dict[int, list[int]] = {}
        for x in members:
            onto = {"main", *((opened[x.pr].branch,) if x.pr in opened and opened[x.pr].branch else ())}
            out = _git(self.repo, "for-each-ref", "--contains", x.head, "--format=%(refname:lstrip=3)", "refs/queue/pr/").stdout
            for n in (int(s) for s in out.split() if s.isdigit()):
                if (n in rest and rest[n].base in onto and refs.get(n) == rest[n].head
                        and f"stacked:{n}<{x.pr}" not in state["told"]):
                    bases.setdefault(n, []).append(x.pr)
        for n, below in sorted(bases.items()):
            try:
                if q.ready(n, rest[n].head):
                    continue
            except QueueError as e:
                q.log(f"research queue vqueue: #{n}'s ready mark can't be read, so it isn't told its base landed now: {e}")
                continue
            names = ", ".join(f"#{b}" for b in below)
            self.tell(state, [f"stacked:{n}<{b}" for b in below],
                      f"#{n}'s base{'s' if len(below) > 1 else ''} {names} landed in main at {tip[:12]}; merge main into #{n}",
                      self._handle(n))

    def beside_args(self, tip: str, base: str) -> tuple[dict, list[str], str]:
        """The run beside ``tip``'s check, ``base`` the commit of `main` below it: (`launch`'s keywords, the steps it skips by name,
        its flags as a message gives them).  Its `--base` is ``base`` (``tip``'s first parent when ``tip`` is ``base``: the run
        beside `main`'s own check), above which ratchet-gate judges the merges; it sends the upstream build when what
        lean-agreement checks changed between them (as check.py finds against that base), and skips lean-agreement else; it
        regenerates no lock record, and reuses the passes a check may."""
        from .merge import required_steps

        kw = {"agreement": "lean-agreement" in required_steps(self.repo, base, tip, "check"), "no_regenerate": True}
        below = (base if base != tip else
                 _git(self.repo, "rev-parse", "--verify", "-q", f"{tip}^1", check=False).stdout.strip()) or None
        flags = (f" --base {below[:12]}" if below else "") + "".join(f" --{k.replace('_', '-')}" for k, on in kw.items() if on)
        return {"base": below, **kw}, ([] if kw["agreement"] else ["lean-agreement"]), flags
