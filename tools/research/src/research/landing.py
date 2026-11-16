"""What the verified queue's loop and its checks share."""
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
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

from . import carry, policy, settings, suite_layout
from .notes import ART_RE
from .queue import CHECK_STEPS, PR, Member, Queue, QueueError, StoreLabels, _git, _who, merge_main, unreviewed
from .reds import CIRCUIT_KILLED, RATCHET_STEP, job_owner

CI = "@ci"
INFRA = "@infra"
TRACE_RE = re.compile(r'^\s*([\w./+-]+\.py):\d+:|File "([^"]+\.py)", line \d+', re.M)
NAMED_RE = re.compile(r"(?<![\w./-])((?:[\w+-]+/)+[\w+.-]*\w\.\w+)")
FAILED_RE = re.compile(r"^FAILED ([^\s:]+)::", re.M)
ERROR_RE = re.compile(r"^ERROR ([^\s:]+\.py)", re.M)
LEAN_AUDIT_RE = re.compile(r"^AUDIT \S*?/lean-audit-scratch-[^/\s]+/(\S+?): FAIL", re.M)
LEAN_ERROR_RE = re.compile(r"^\s*error: (\S+?\.lean):\d+", re.M)
LEAN_ORPHAN_RE = re.compile(r"^\s+orphan: (\S+?\.lean) is not imported")
LEAN_LAYERS_RE = re.compile(r"^\s+layers: ([A-Za-z][\w.]*) imports ")
KILLED_RC = (129, 137, 143)
KILLED_SAYS = "not a kernel verdict"
CIRCUIT_FAIL_RE = re.compile(r"^FAIL (\S+)", re.M)
ACTIVE = ("submitted", "waiting", "running", "unknown")
#: check.py's exit for a twin that never took a slot
NOT_STARTED = 75
AFTER_LANDING = tuple(s for s in CHECK_STEPS if not s.startswith("preflight-"))
AFTER_LANDING_OPTIONAL = ("lean-agreement",)
SUITE_STEPS = ("lean-suites", "pytest")
SLIM_STEPS = ("lean-agreement", RATCHET_STEP)
SLIM_JUDGED = (*SLIM_STEPS, *(s for s in CHECK_STEPS if s.startswith("preflight-")))
LOST_AFTER_S = 3 * 3600
CIRCUIT_RED = "step circuit-check"
JOB_OF_KIND = {"test-file": "pytest", "lean-module": "lean-audit", "lean-lock": "lean-audit", "circuit-binding": "circuit-check",
               "circuit-coverage": "circuit-check", "circuit-catalog": "circuit-check"}
SUITES_KIND = "lean-suites"
CIRCUIT_NAMED = 100
LEAN_STEP = "lean-audit"
SIGTERM_RC = 143
RUN_RE = re.compile(r"^RUN ((?:r20\d{6}-\d{6}-[0-9a-f]{4} ?)+)$", re.M)


def auth_env(auth: tuple[str, ...]) -> dict[str, str]:
    pairs = [o.split("=", 1) for flag, o in zip(auth[::2], auth[1::2]) if flag == "-c"]
    n = int(settings.get("GIT_CONFIG_COUNT") or 0)
    env = {"GIT_CONFIG_COUNT": str(n + len(pairs))}
    for i, (k, v) in enumerate(pairs, n):
        env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"] = k, v
    return env if pairs else {}


@contextlib.contextmanager
def authed(auth: tuple[str, ...]):
    extra = {**auth_env(auth), "GIT_TERMINAL_PROMPT": "0"}
    was = {k: settings.named(k) for k in extra}
    os.environ.update(extra)
    try:
        yield
    finally:
        for k, v in was.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


OWN_TRAIN = Path(__file__).resolve().parents[3] / "check" / "train.sh"
TRAIN_COMPANIONS = ("settings.py",)


def train_launch(repo: Path, commit: str, machine: str, agreement: bool, auth: tuple[str, ...] = (),
                 deferred: str | None = None, no_cache: bool = False, no_regenerate: bool = False,
                 detached: bool = False, circuit_cache: bool = False, shard: str | None = None,
                 base: str | None = None, cc_records: bool = False,
                 cc_records_from: str | None = None, shard_plan: str | None = None,
                 ratchet_base: str | None = None, rank: int | None = None,
                 bookkeeping_warn: bool = False, script_from: Path | None = None,
                 deselect: str | None = None) -> tuple[list[str], str] | dict:
    argv = ["bash", "tools/check/train.sh", commit, machine, *(["--cc-records"] if cc_records else [])]
    if not cc_records:
        argv += [*(["--agreement"] if agreement else []),
                 *(["--deferred", deferred] if deferred else []), *(["--no-cache"] if no_cache else []),
                 *(["--circuit-cache"] if circuit_cache else []), *(["--no-regenerate"] if no_regenerate else []),
                 *(["--bookkeeping-warn"] if bookkeeping_warn else []),
                 *(["--base", base] if base else []), *(["--ratchet-base", ratchet_base] if ratchet_base else []),
                 *(["--shard", shard] if shard else []),
                 *(["--shard-plan-file", shard_plan] if shard_plan else []),
                 *(["--deselect", deselect] if deselect else []),
                 *(["--cc-records-from", cc_records_from] if cc_records_from else [])]
    env = {**os.environ, **auth_env(auth), "GIT_TERMINAL_PROMPT": "0", "TRAIN_AS": "ci",
           "CHECK_SLOT_RANK": "" if rank is None else str(rank)}
    if detached:
        return detach(repo, argv, env, script=argv[1], source=script_from, companions=TRAIN_COMPANIONS)
    r = subprocess.run(argv, cwd=repo, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    m = RUN_RE.search(r.stdout)
    return (m[1].split() if m else []), (r.stdout + r.stderr).strip()[-600:]


AFTER_LANDING_SLOT_RANK = 0
LAUNCHES = "drive-launches"
LAUNCH_LOST_S = 5400
DETACH_JOB = 'd=$1; shift; "$@" > "$d/log" 2>&1; echo "$?" > "$d/rc.tmp" && mv "$d/rc.tmp" "$d/rc"'


def detach(repo: Path, argv: list[str], env: dict[str, str], script: str | None = None, source: Path | None = None,
           companions: tuple[str, ...] = ()) -> dict:
    root = Path(tempfile.gettempdir()) / LAUNCHES
    root.mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(prefix="train-", dir=root)
    if script:
        src = Path(source or repo / script)
        copy = shutil.copy(src, Path(d) / Path(script).name)
        for name in companions:
            with contextlib.suppress(FileNotFoundError):
                shutil.copy(src.parent / name, Path(d) / name)
        argv = [str(copy) if a == script else a for a in argv]
    p = subprocess.Popen(["sh", "-c", DETACH_JOB, "sh", d, *argv], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"dir": d, "pid": p.pid}


QUICK_JOB = ('t=$1 c=$2 rid=$3; shift 3; git worktree add -q --detach "$t/tree" "$c" || exit; (cd "$t/tree" && "$@"); rc=$?; '
             'git worktree remove --force "$t/tree"; rm -rf "$t"; git worktree prune; [ "$rc" = 0 ] && echo "RUN $rid"; exit "$rc"')
QUICK_WAIT_S, QUICK_RUN_S = 3600, 3 * 3600


def quick_launch(repo: Path, commit: str, base: str, machine: str, auth: tuple[str, ...] = ()) -> dict:
    from . import runs as R
    from .queue import quick_run

    root = Path(tempfile.gettempdir()) / LAUNCHES
    root.mkdir(parents=True, exist_ok=True)
    t = tempfile.mkdtemp(prefix="quick-", dir=root)
    rid = R.new_run_id()
    argv = quick_run(rid, machine, Path(t) / "tree", commit, base, by="ci", wait_s=QUICK_WAIT_S, timeout_s=QUICK_RUN_S,
                     priority=True)
    env = {**os.environ, **auth_env(auth), "GIT_TERMINAL_PROMPT": "0"}
    return detach(repo, ["sh", "-c", QUICK_JOB, "sh", t, commit, rid, *argv], env)


def _epoch(ts: object) -> float | None:
    """Seconds since the epoch; None when it isn't a `YYYY-MM-DDTHH:MM:SSZ` timestamp."""
    if ts is None or ts == "":
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError as e:
        print(f"landing: {str(ts)[:40]!r} is not a time, read as none ({e})", file=sys.stderr, flush=True)
        return None
    return (t if t.tzinfo else t.replace(tzinfo=UTC)).timestamp()


def _job_end(rec: dict, name: str) -> tuple[str, str, str | None] | str | None:
    d = Path(rec["dir"])
    if not (d / "rc").is_file():
        try:
            os.kill(int(rec["pid"]), 0)
        except ProcessLookupError:
            return "its process is gone with no exit written (a restart of the pod)"
        except (PermissionError, ValueError, TypeError, KeyError):
            pass
        return None
    try:
        rc = (d / "rc").read_text().strip()
        tail = (d / "log").read_text(errors="replace").strip()[-600:]
        text = (Path(rec["out"]) / name).read_text() if rc == "0" else None
    except OSError as e:
        rc, tail, text = "?", f"{type(e).__name__}: {e}", None
    shutil.rmtree(d, ignore_errors=True)
    return rc, tail, text


def shard_machines(on: str, n: int) -> list[str]:
    ms = on.split(",")
    out = []
    for k in range(n):
        own = ms[k::n] if len(ms) >= n else [ms[k % len(ms)]]
        r = k % len(ms)
        out.append(",".join([*own, *(m for m in ms[r:] + ms[:r] if m not in own)]))
    return out


def elsewhere(on: str, ran: Iterable[str]) -> str:
    ran = set(ran)
    return ",".join(m for m in on.split(",") if m not in ran) or on


def _hhmm(t: float) -> str:
    return f"{datetime.fromtimestamp(t, UTC):%H:%M} UTC"


def launched(launch: dict) -> tuple[list[str] | None, str]:
    d = Path(launch["dir"])
    try:
        out = (d / "log").read_text(errors="replace")
    except OSError:
        out = ""
    tail = out.strip()[-600:]
    if m := RUN_RE.search(out[: out.rfind("\n") + 1]):
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


PLAN_WAIT_S = 2400
AUTO_PLAN = "check-shard-plan.json"
PLAN_JOB = ('set -e; d=$1 r=$PWD; t="${TRAIN_TREES:-${1%/*}}/plan-${1##*/}"; mkdir -p "${t%/*}"; '
            'export UV_PROJECT_ENVIRONMENT="${t%/*}/plan-venv"; git worktree add -q --detach "$t" "$2"; '
            'trap \'s=$?; set +e; cd "$r"; git worktree remove --force "$t" 2>/dev/null || rm -rf "$t" 2>/dev/null || '
            'rm -rf "$t"; git worktree prune; exit $s\' EXIT; cd "$t"; shift 2; '
            'uv run -q python tools/check/check.py --shard-plan --out "$d/' + AUTO_PLAN + '" "$@"')


def plan_shards(repo: Path, tip: str, steps: str, max_shards: int | None = None, n: int | None = None) -> dict:
    root = Path(tempfile.gettempdir()) / LAUNCHES
    root.mkdir(parents=True, exist_ok=True)
    out = tempfile.mkdtemp(prefix="plan-", dir=root)
    args = ["--deferred", steps, *(["--max-shards", str(max_shards)] if max_shards else []), *(["--n", str(n)] if n else [])]
    return {**detach(repo, ["sh", "-c", PLAN_JOB, "sh", out, tip, *args], {**os.environ, "GIT_TERMINAL_PROMPT": "0"}),
            "out": out}


def planned(rec: dict) -> tuple[dict | None, str | None]:
    if (end := _job_end(rec, AUTO_PLAN)) is None or isinstance(end, str):
        return None, end
    rc, tail, text = end
    try:
        doc = json.loads(text) if text is not None else None
    except ValueError as e:
        doc, tail = None, f"{type(e).__name__}: {e}"
    if isinstance(doc, dict) and doc.get("kind") == "check-shard-plan/v1" and isinstance(doc.get("n"), int) and doc["n"] >= 1:
        return doc, None
    return None, f"`check.py --shard-plan` exited {rc}: {tail}"


PROGRESS = ("steps.jsonl", "suites.failed.json", "suites-lean.failed.json")
SPANS = "spans.jsonl"
LIVE = "check-live.json"
PREFLIGHT = tuple(s for s in CHECK_STEPS if s.startswith("preflight-"))
PREBUILD_STEP = "prebuild"


def failing(run_dir: Path) -> str | None:
    out = []
    try:
        lines = (run_dir / PROGRESS[0]).read_text().splitlines() if (run_dir / PROGRESS[0]).exists() else []
    except OSError as e:
        print(f"landing: {run_dir / PROGRESS[0]} unread ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        lines = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError as e:
            print(f"landing: a line of {PROGRESS[0]} is not JSON, skipped ({e})", file=sys.stderr, flush=True)
            continue
        if rec.get("status") == "failed":
            out.append(f"step {rec.get('step')}")
    for name in PROGRESS[1:]:
        if not (run_dir / name).exists():
            continue
        try:
            doc = json.loads((run_dir / name).read_text())
        except (OSError, ValueError) as e:
            print(f"landing: {run_dir / name} unread, skipped ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            continue
        for s in doc if isinstance(doc, list) else ():
            tests = s.get("failed_tests") or []
            out.append(f"suite {s.get('name')}" + (f" ({', '.join(tests[:2])}{' ...' if len(tests) > 2 else ''})" if tests else ""))
    return "; ".join(out) or None


def gating(repo: Path, commit: str) -> list[str]:
    from .merge import declared_jobs

    jobs = declared_jobs(repo, commit)
    return [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL and (isinstance(jobs, str) or s in jobs)]


def run_states(repo: Path, ids: list[str]) -> dict[str, tuple]:
    return Reads().states(repo, ids)


STATUS_JOBS = 8
NODE_JOBS = 2
EVIDENCE = (*PROGRESS, SPANS, LIVE, "suites.json", "suites-lean.json")
FINAL = ".landing-final.json"


def _ended(run_dir: Path) -> bool:
    from .remote import TERMINAL, _read_json

    return (_read_json(run_dir / "status.json") or {}).get("state") in TERMINAL


def _final(run_dir: Path) -> frozenset[str]:
    if not _ended(run_dir) or not (marker := run_dir / FINAL).is_file():
        return frozenset()
    names = json.loads(marker.read_text()).get("names")
    return frozenset(n for n in names if isinstance(n, str)) if isinstance(names, list) else frozenset()


def read_run(rid: str, names: tuple[str, ...], root: Path | None = None) -> tuple[Path, dict] | None:
    from . import remote
    from . import runs as R
    from .checkview import CHECK_FILES
    from .cli import _run_dir

    also = tuple(dict.fromkeys((*names, *CHECK_FILES)))

    def fetched(d: Path) -> bool:
        try:
            return "last_fetch_error" not in remote.fetch(d, also=also)
        except remote.Refused as e:
            print(f"research status: {rid}: {e}", file=sys.stderr)
            return False

    try:
        d = _run_dir(R.runs_root(root), rid, None)
        final = _final(d)
        if (remote._read_json(d / "remote.json") or {}).get("remote_run_dir") and not set(names) <= final:
            ended = _ended(d)
            ok = fetched(d)
            if ok and not ended and _ended(d):
                ok, ended = fetched(d), True
            if ok and ended:
                part = d / f"{FINAL}.{os.getpid()}.{threading.get_ident()}"
                part.write_text(json.dumps({"names": sorted(final | set(names))}))
                os.replace(part, d / FINAL)
        return d, R.summarize(d)
    except Exception as e:  # noqa: BLE001  (`research status` exits without a row: the run reads as unknown, as it did)
        print(f"research status: {rid}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _node(rid: str) -> str:
    from . import runs as R
    from .remote import _read_json

    return str((_read_json(R.runs_root(None) / rid / "remote.json") or {}).get("machine") or rid)


def _spread(fn, items: list, node: Callable[[object], str], jobs: int = STATUS_JOBS, node_jobs: int = NODE_JOBS) -> list:
    items = list(items)
    if len(items) < 2:
        return [fn(x) for x in items]
    out, keys, left = [None] * len(items), [node(x) for x in items], list(range(len(items)))
    busy: dict[str, int] = {}
    errors: list[BaseException] = []
    cv = threading.Condition()

    def one(i: int) -> None:
        try:
            out[i] = fn(items[i])
        except BaseException as e:  # noqa: BLE001  (raised in the caller once every read ended)
            errors.append(e)
        finally:
            with cv:
                busy[keys[i]] -= 1
                cv.notify()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, len(items))) as ex, cv:
        while left:
            i = next((i for i in left if busy.get(keys[i], 0) < node_jobs), None)
            if i is None or sum(busy.values()) >= jobs:
                cv.wait()
                continue
            left.remove(i)
            busy[keys[i]] = busy.get(keys[i], 0) + 1
            ex.submit(one, i)
    if errors:
        raise errors[0]
    return out


class Reads:
    def __init__(self, read: Callable[[str, tuple[str, ...]], tuple[Path, dict] | None] = read_run,
                 node: Callable[[str], str] = _node, jobs: int = STATUS_JOBS, node_jobs: int = NODE_JOBS):
        self.read, self.node, self.jobs, self.node_jobs = read, node, jobs, node_jobs
        self._got: dict[str, tuple[frozenset[str], tuple[Path, dict] | None]] | None = None

    def __call__(self, asks: list[tuple[str, tuple[str, ...]]]) -> list[tuple[Path, dict] | None]:
        if self._got is None:
            return _spread(lambda a: self.read(*a), asks, lambda a: self.node(a[0]), self.jobs, self.node_jobs)
        need: dict[str, tuple[str, ...]] = {}
        for rid, names in asks:
            if rid not in self._got or not set(names) <= self._got[rid][0]:
                need[rid] = tuple(dict.fromkeys((*need.get(rid, ()), *names)))
        todo = list(need.items())
        for (rid, names), got in zip(todo, _spread(lambda a: self.read(*a), todo, lambda a: self.node(a[0]), self.jobs,
                                                   self.node_jobs)):
            had, was = self._got.get(rid, (frozenset(), None))
            self._got[rid] = (had | set(names), got if got is not None else was)
        return [self._got[rid][1] for rid, _ in asks]

    @contextlib.contextmanager
    def tick(self, ids: list[str] = ()):
        self._got = {}
        try:
            self.evidence(Path("."), list(dict.fromkeys(ids)))
            yield self
        finally:
            self._got = None

    def states(self, repo: Path, ids: list[str]) -> dict[str, tuple]:
        def one(got: tuple[Path, dict] | None) -> tuple:
            try:
                run = got[1]
                state = run.get("state") or "unknown"
                return (state, run.get("rc"), run.get("ended_utc"),
                        failing(Path(run["dir"])) if state in ACTIVE and run.get("dir") else None,
                        (run.get("remote") or {}).get("machine"))
            except (TypeError, KeyError, AttributeError):
                return ("unknown", None, None, None)

        return dict(zip(ids, map(one, self([(r, (*PROGRESS, LIVE)) for r in ids]))))

    def evidence(self, repo: Path, ids: list[str]) -> list[Path]:
        dirs = {r: Path(got[1]["dir"]) for r, got in zip(ids, self([(r, EVIDENCE) for r in ids]))
                if got is not None and isinstance(got[1], dict) and got[1].get("dir")}
        logs = {r: [f"suites/{Path(s['log']).stem}{x}" for s in failed_suites(d) if s.get("log")
                    for x in (".log", *((".json", ".reads") if (s.get("suite") or ".") != "." else ()))]
                + [Path(st["log"]).name for st in failed_steps(d) if st.get("log") and st.get("step") not in SUITE_STEPS]
                for r, d in dirs.items()}
        self([(r, tuple(names)) for r, names in logs.items() if names])
        return [dirs[r] for r in ids if r in dirs]


def cancel_runs(repo: Path, ids: list[str], reason: str) -> dict[str, str]:
    failed = {}
    for rid in ids:
        r = subprocess.run([sys.executable, "-m", "research", "cancel", rid, "--by", "ci", "--reason", reason], cwd=repo,
                           capture_output=True, text=True)
        if r.returncode:
            failed[rid] = (r.stderr or r.stdout).strip()[-300:] or f"exit {r.returncode}"
    return failed


def merge_thread_post(text: str, to: tuple[str, ...] = (CI,), sender: str = "ci") -> bool:
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
        if not (run_dir / name).exists():
            continue
        try:
            doc = json.loads((run_dir / name).read_text())
        except (OSError, ValueError) as e:
            print(f"landing: {run_dir / name} unread, skipped ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            continue
        out += [x for x in doc if isinstance(x, dict)] if isinstance(doc, list) else [doc] if isinstance(doc, dict) else []
    return out


def failed_suites(run_dir: Path) -> list[dict]:
    done = [s for s in _docs(run_dir, "suites.json", "suites-lean.json") if s.get("status") == "failed"]
    return done or _docs(run_dir, *PROGRESS[1:])


def evidence(repo: Path, ids: list[str]) -> list[Path]:
    return Reads().evidence(repo, ids)


def failed_steps(run_dir: Path) -> list[dict]:
    out = []
    with contextlib.suppress(OSError):
        for line in (run_dir / PROGRESS[0]).read_text().splitlines():
            with contextlib.suppress(ValueError):
                if (rec := json.loads(line)).get("status") == "failed":
                    out.append(rec)
    return out


def suite_pyproject(repo: Path, commit: str, suite: str) -> dict:
    import tomllib

    own = "pyproject.toml" if suite.strip("/") in ("", ".") else f"{suite.strip('/')}/pyproject.toml"
    r = _git(repo, "show", f"{commit}:{own}", check=False)
    try:
        return tomllib.loads(r.stdout) if not r.returncode else {}
    except tomllib.TOMLDecodeError as e:
        print(f"landing: {own} at {commit[:12]} did not parse, read as empty ({e})", file=sys.stderr, flush=True)
        return {}


@functools.lru_cache(maxsize=16)
def member_dirs(repo: Path, commit: str) -> dict[str, str]:
    return suite_layout.member_dirs(lambda d: suite_pyproject(repo, commit, d))


def suite_name(repo: Path, commit: str, suite: str) -> str:
    path = suite.strip("/") or "."
    if path == ".":
        return suite_layout.ROOT_SUITE
    return suite_layout.suite_name(path, suite_pyproject(repo, commit, path).get("project", {}), member_dirs(repo, commit))


def lean_errors(log: str) -> set[str]:
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
            failures = True
        elif pkg and failures and (m := re.match(r"^\s*- ([A-Za-z][\w.]*)\s*$", line)):
            parts = m[1].split(".")
            root = os.path.dirname(pkg) if parts[0] == os.path.basename(pkg) else pkg
            out.add(os.path.normpath(os.path.join(root, *parts) + ".lean"))
        elif failures and not line.strip().startswith("-"):
            failures = False
    return out


def killed(run_dir: Path, why: str = "") -> bool:
    steps = failed_steps(run_dir)
    if any(isinstance(rc, int) and (rc < 0 or rc in KILLED_RC) for rc in (st.get("rc") for st in steps)):
        return True
    for st in steps:
        if st.get("log"):
            with contextlib.suppress(OSError):
                if KILLED_SAYS in (run_dir / Path(st["log"]).name).read_text(errors="replace"):
                    return True
    if len(steps) == 1 and steps[0].get("step") == "circuit-check" and steps[0].get("log"):
        with contextlib.suppress(OSError):
            fails = CIRCUIT_FAIL_RE.findall((run_dir / Path(steps[0]["log"]).name).read_text(errors="replace"))
            if fails and all(f == CIRCUIT_KILLED for f in fails):
                return True
    return any(f"rc={rc}" in why for rc in (*KILLED_RC, -9, -15))


def tree_root(run_dir: Path) -> str | None:
    if not (job := run_dir / "job.json").exists():
        return None
    cwd = (json.loads(job.read_text()).get("run") or {}).get("cwd")
    return cwd.rstrip("/") if isinstance(cwd, str) and cwd.startswith("/") else None


def relative(path: str, base: str = "", root: str | None = None) -> str | None:
    if path.startswith("/"):
        if root is None or not path.startswith(root + "/"):
            return None
        path, base = path[len(root) + 1:], ""
    path = os.path.normpath(os.path.join(base, path))
    return None if path in (".", "..") or path.startswith(("../", ".venv/")) else path


def _report(log: str) -> str:
    start = min((i for i in (log.find("= ERRORS ="), log.find("= FAILURES =")) if i >= 0), default=-1)
    if start < 0:
        return log
    ends = [i for i in (log.find("= warnings summary =", start), log.find("short test summary info", start)) if i >= 0]
    return log[start:min(ends, default=len(log))]


def _named(log: str, base: str, root: str | None = None) -> set[str]:
    report = _report(log)
    paths = {relative(m[1] or m[2], base, root) for m in TRACE_RE.finditer(report)}
    paths |= {relative(m[1], base, root) for m in NAMED_RE.finditer(report)}
    return {p for p in paths if p} | {f"art:{h}" for h in ART_RE.findall(report)}


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
    return step_named(st.get("step"), _step_log(run_dir, st), tree_root(run_dir))


def step_named(step: str | None, log: str, root: str | None = None) -> set[str] | None:
    if step == "lean-audit" and (lean := lean_errors(log)):
        return lean
    if not (tests := FAILED_RE.findall(log)):
        return None
    return {os.path.normpath(t) for t in tests} | _named(log, "", root)


def suite_paths(run_dir: Path, s: dict) -> set[str] | None:
    return suite_named(s, _suite_log(run_dir, s), tree_root(run_dir))


def suite_named(s: dict, log: str, root: str | None = None) -> set[str] | None:
    base, tests = s.get("suite") or "", s.get("failed_tests") or []
    base = "" if base == "." else base
    if not tests and (errors := ERROR_RE.findall(log)):
        tests = errors
    if not tests:
        out, child = s.get("outside") or {}, s.get("child_outside") or {}
        modules = list(out.values() if isinstance(out, dict) else ()) + list(child)
        files = list(out) + [f for fs in child.values() for f in fs]
        if not files:
            return None
        return {os.path.normpath(p) for p in files} | {os.path.normpath(m) for m in modules if m and m != "*"}
    paths = {os.path.normpath(os.path.join(base, t.split("::")[0])) for t in tests}
    return paths | (_named(log, base, root) if log else set())


def implicated(run_dir: Path) -> set[str] | None:
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


SUITE_REPORTS = {"suites.failed.json": "pytest", "suites-lean.failed.json": "lean-suites", "suites.json": "pytest",
                 "suites-lean.json": "lean-suites"}


def suite_rows(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, step in SUITE_REPORTS.items():
        for s in _docs(run_dir, name):
            if n := s.get("name") or s.get("suite"):
                out[n] = {**out.get(n, {}), **s, "step": step, "status": s.get("status") or "failed"}
    return out


def unread_suites(run_dir: Path) -> set[str]:
    from .remote import _read_json

    if not (_read_json(run_dir / "remote.json") or {}).get("remote_run_dir"):
        return set()
    rows = {row["step"] for row in suite_rows(run_dir).values() if row["status"] == "failed"}
    final = _final(run_dir)
    return {s for st in failed_steps(run_dir) if (s := st.get("step")) in SUITE_STEPS and s not in rows and not not_run(st)
            and not {n for n, x in SUITE_REPORTS.items() if x == s} <= final}


def step_records(run_dir: Path) -> list[dict]:
    out = []
    with contextlib.suppress(OSError):
        for line in (run_dir / PROGRESS[0]).read_text().splitlines():
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
    return out


def live_steps(run_dir: Path) -> list[dict]:
    doc = next(iter(_docs(run_dir, LIVE)), {})
    over, steps = doc.get("state") in ("passed", "failed", "stopped"), doc.get("steps")
    return [{**s, "state": "stopped"} if over and s.get("state") in ("pending", "running") else s
            for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


RANK = {"passed": 4, None: 3, "failed": 2, "skipped": 1, "stopped": 0}


def shard_steps(groups: list[list[Path]], done: list[bool]) -> list[dict[str, str | None]] | None:
    out = []
    for dirs, known in zip(groups, done):
        got: dict[str, str | None] = {}
        for d in dirs:
            live = live_steps(d)
            known = known or any(s.get("step") not in (*PREFLIGHT, PREBUILD_STEP) for s in live)
            mine = {s.get("step") or "": s.get("state") for s in live}
            mine.update({st.get("step") or "": "stopped" if not_run(st) else st.get("status") or "failed" for st in step_records(d)})
            for step, v in mine.items():
                v = None if v in ("pending", "running", None) else v
                if step not in got or RANK.get(v, RANK["failed"]) > RANK.get(got[step], RANK["failed"]):
                    got[step] = v
        if len(groups) > 1 and not known:
            return None
        out.append(got)
    return out


def shard_passed(view: list[dict[str, str | None]] | None, allowed: set[str] = frozenset()) -> set[str]:
    def ok(s: str, x: str | None) -> bool:
        return x == "passed" or (x == "failed" and s in allowed)
    return {s for v in view or () for s, x in v.items() if ok(s, x) and all(ok(s, w[s]) or w[s] == "skipped" for w in view if s in w)}


def not_run(st: dict) -> bool:
    return st.get("status") == "stopped" or (st.get("status") == "failed" and st.get("rc") in (-15, SIGTERM_RC))


def failed_names(dirs: list[Path]) -> set[str]:
    return {st.get("step") for d in dirs for st in step_records(d) if st.get("status") == "failed" and not not_run(st)}


def allowed(pol: policy.Policy, now: float, key: str, d: Path, rec: dict) -> policy.Coverage:
    if key.startswith("suite "):
        return pol.suite(key.split(" ", 1)[1], rec, now)
    if key == CIRCUIT_RED:
        return pol.circuit(_step_log(d, rec), now)
    return policy.Coverage(False)


def allowed_steps(pol: policy.Policy, now: float, dirs: list[Path]) -> set[str]:
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
                covered = bool(failed) and all(allowed(pol, now, f"suite {n}", d, s).covered for n, s in failed)
            else:
                covered = allowed(pol, now, f"step {step}", d, st).covered
            (ok if covered else bad).add(step)
    return ok - bad


class Landing:
    def __init__(self, q: Queue, *, on: str,
                 launch: Callable[[Path, str, str, bool], tuple[list[str], str]] = train_launch,
                 states: Callable[[Path, list[str]], dict] | None = None,
                 post: Callable[[str, tuple[str, ...]], bool] = merge_thread_post,
                 now: Callable[[], float] = time.time, cancel: Callable[[Path, list[str], str], dict[str, str] | None] = cancel_runs,
                 evidence: Callable[[Path, list[str]], list[Path]] | None = None, warn_steps: tuple[str, ...] = (),
                 reviews: Callable[[int, str], list] | None = None, ready_by: Callable[[int, str], str | None] | None = None):
        self.q, self.on = q, on
        self.reads = Reads()
        states, evidence = states or self.reads.states, evidence or self.reads.evidence
        self.warn_steps = warn_steps
        self.reviews: Callable[[int, str], list] = reviews or (lambda pr, head: StoreLabels(q.store, q.log).reviews(pr, head))
        self.ready_by: Callable[[int, str], str | None] = ready_by or (lambda pr, head: StoreLabels(q.store, q.log).ready_by(pr, head))
        self.delta: Callable[[str, str, list[str], set[str]], carry.Delta] = (
            lambda old, new, landed, trusted: carry.delta(self.repo, old, new, carry.trust(self.repo, landed, trusted)))
        self.plan_shards: Callable[..., dict] = plan_shards
        self.planned: Callable[[dict], tuple[dict | None, str | None]] = planned
        self.launch_after_landing: Callable[..., tuple[list[str], str] | dict] = (
            (lambda repo, commit, machine, steps, agreement=False, no_cache=False, circuit_cache=False, no_regenerate=False,
                    shard=None, base=None, cc_records=False, cc_records_from=None, shard_plan=None, ratchet_base=None,
                    train_from=None, deselect=None:
             train_launch(repo, commit, machine, agreement, auth=q.auth, deferred=steps, no_cache=no_cache,
                          no_regenerate=no_regenerate, detached=True, circuit_cache=circuit_cache, shard=shard, base=base,
                          cc_records=cc_records, cc_records_from=cc_records_from, shard_plan=shard_plan,
                          ratchet_base=ratchet_base, rank=AFTER_LANDING_SLOT_RANK, script_from=train_from, deselect=deselect))
            if launch is train_launch else (lambda repo, commit, machine, steps, **_: self.launch(repo, commit, machine, False)))
        self.launch_locks: Callable[[Path, str, str], tuple[list[str], str] | dict] = (
            (lambda repo, commit, machine: train_launch(repo, commit, machine, False, auth=q.auth, deferred=LEAN_STEP,
                                                        detached=True, rank=AFTER_LANDING_SLOT_RANK, bookkeeping_warn=True,
                                                        script_from=OWN_TRAIN))
            if launch is train_launch else (lambda repo, commit, machine: self.launch(repo, commit, machine, False)))
        self.launch_quick: Callable[[Path, str, str, str], tuple[list[str], str] | dict] = (
            lambda repo, commit, base, machine: quick_launch(repo, commit, base, machine, auth=q.auth))
        self.launched: Callable[[dict], tuple[list[str] | None, str]] = launched
        if launch is train_launch:
            launch = functools.partial(train_launch, auth=q.auth)
        self.launch, self.states, self.post, self.now, self.cancel, self.evidence = launch, states, post, now, cancel, evidence
        self.repo, self.dry_run, self._prs, self._ev = q.workdir, False, None, {}

    def reading(self, ids: list[str]):
        if self.states != self.reads.states and self.evidence != self.reads.evidence:
            return contextlib.nullcontext()
        return self.reads.tick(ids)

    def _fetched(self, attempt: str) -> list[Path]:
        if attempt not in self._ev:
            self._ev[attempt] = self.evidence(self.repo, attempt.split())
        return self._ev[attempt]

    @staticmethod
    def _started(got: tuple[list[str], str] | dict) -> tuple[dict | None, str]:
        if isinstance(got, dict):
            return {"ids": [], "launch": got}, ""
        ids, tail = got
        return ({"ids": ids} if ids else None), tail

    def _resolve(self, rec: dict, what: str) -> str | None:
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

    @staticmethod
    def _job(task: dict) -> str:
        return JOB_OF_KIND.get(task["kind"], task["kind"])

    def _suite_name(self, tip: str, suite: str) -> str:
        cache = self.__dict__.setdefault("_suite_names", {})
        if (tip, suite) not in cache:
            cache[tip, suite] = suite_name(self.repo, tip, suite)
        return cache[tip, suite]

    def _task_unit(self, tip: str, task: dict) -> str:
        if task["kind"] == "test-file":
            return f"suite {self._suite_name(tip, (task.get('vars') or {}).get('suite') or '.')}"
        return f"step {self._job(task)}"

    def _task_units(self, tip: str, task: dict, result: dict) -> dict[str, dict | None]:
        rows = (result.get("detail") or {}).get("failed") if task["kind"] == SUITES_KIND else None
        named = {f"suite {r['name']}": r for r in rows or () if isinstance(r, dict) and isinstance(r.get("name"), str)}
        return named or {self._task_unit(tip, task): None}

    def _task_allowed(self, tip: str, task: dict, result: dict, row: dict | None = None,
                      at: str | None = None) -> policy.Coverage:
        pol, now, d = self._policy(at or tip), self.now(), result.get("detail") or {}
        if row is not None:
            return pol.suite(str(row["name"]), row, now)
        if task["kind"] == "test-file":
            tests = [str(x) for x in d.get("failed_tests") or ()]
            row = {"failed_tests": tests, "beyond_tests": not (d.get("exit") == 1 and bool(tests))}
            return pol.suite(self._task_unit(tip, task).split(" ", 1)[1], row, now)
        if self._job(task) == "circuit-check" and len(names := [str(x) for x in d.get("failures") or ()]) < CIRCUIT_NAMED:
            return pol.circuit("".join(f"FAIL {n}\n" for n in names), now)
        return policy.Coverage(False)

    def _gating(self, commit: str) -> list[str]:
        cache = self.__dict__.setdefault("_gating_of", {})
        if commit not in cache:
            cache[commit] = gating(self.repo, commit)
        return cache[commit]

    def _shards(self, run: dict) -> list[list[Path]]:
        dirs = self._fetched(" ".join(run["ids"]))
        return [[d for d in dirs if d.name in g] for g in run["groups"]] if run.get("groups") else [dirs]

    @staticmethod
    def _every_shard(run: dict, states: dict, ok: Callable[[object], bool]) -> bool:
        return all(any(ok(states[r][1]) for r in g if r in states) for g in run.get("groups") or [run["ids"]])

    def _never_ran(self, run: dict, states: dict) -> bool:
        never = (NOT_STARTED, SIGTERM_RC)
        groups = run.get("groups") or [run["ids"]]
        missed = [g for g in groups if all(states[r][1] in never for r in g if r in states)]
        return bool(missed) and all(g in missed or any(states[r][1] == 0 for r in g if r in states) for g in groups)

    def _unrun(self, ids: list[str], steps: list[str], tip: str | None = None) -> list[str]:
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
        cache = self.__dict__.setdefault("_policy_of", {})
        if commit not in cache:
            got = policy.at(self.repo, commit)
            if isinstance(got, str):
                self.q.log(f"research queue drive: {got}: no failure is allowed at {commit[:12]}")
                got = policy.Policy()
            cache[commit] = got
        return cache[commit]

    def _allowed(self, tip: str, key: str, d: Path, rec: dict) -> policy.Coverage:
        return allowed(self._policy(tip), self.now(), key, d, rec)

    def _allowed_steps(self, tip: str, dirs: list[Path]) -> set[str]:
        return allowed_steps(self._policy(tip), self.now(), dirs)

    def _only_allowed(self, run: dict, at: str | None = None) -> bool:
        dirs = [d for ds in self._shards(run) for d in ds]
        failed = failed_names(dirs)
        return bool(failed) and failed <= self._allowed_steps(at or run["tip"], dirs)

    def _lost(self, run: dict, states: dict) -> bool:
        live = [st for st, *_ in states.values() if st in ACTIVE]
        return bool(live) and all(st == "unknown" for st in live) and self.now() - run.get("at", self.now()) > LOST_AFTER_S

    def _path_owner(self, path: str) -> str | None:
        from .queue import REGISTRY
        from .slack import owner

        if (reg := self.__dict__.get("_reg")) is None:
            r = _git(self.repo, "show", f"origin/main:{REGISTRY}", check=False)
            try:
                reg = json.loads(r.stdout) if not r.returncode else {}
            except ValueError as e:
                print(f"landing: origin/main's {REGISTRY} is not JSON, so no handle owns a path ({e})", file=sys.stderr, flush=True)
                reg = {}
            self._reg = reg = reg if isinstance(reg, dict) and isinstance(reg.get("handles"), list) else {"handles": []}
        try:
            return owner(reg, path)
        except (KeyError, TypeError, AttributeError):
            return None

    def fetch(self, rid: str, names: list[str]) -> tuple[Path, dict] | None:
        from .merge import fetch_run

        return fetch_run(rid, names, self.repo)

    def _on_main(self, state: dict, members: list[Member], since: str, tip: str, attempt: str) -> None:
        from .merge import _sweep_after, caught_up

        q = self.q
        self._stacked(state, members, tip)

        with authed(q.auth):
            caught_up(self.repo, "origin", "check", since=since, log=q.log,
                      runs={x.commit: attempt for x in members} | {tip: attempt})
            _sweep_after(self.repo, "main", "origin", keep=state.get("stale_closed", {}))

    def _jobs(self, commit: str) -> list[dict]:
        from .merge import declared

        cache = self.__dict__.setdefault("_jobs_at", {})
        if commit not in cache:
            cache[commit] = declared(self.repo, commit) or []
        return cache[commit]

    def _step_owner(self, step: str, commit: str) -> str | None:
        return job_owner(self._jobs(commit), step)

    def _handle(self, pr: int) -> str | None:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self.q.handle(self._prs[pr]) if pr in self._prs else None

    def _owner(self, m: Member) -> str | None:
        return self._handle(m.pr) or self.q.handle(PR(m.pr, m.head, body=f"Lane {m.owner}"))

    def _owners(self, m: Member) -> set[str | None]:
        return {m.owner, self._owner(m), self.ready_by(m.pr, m.head)}

    def _carried_from(self, m: Member, heads: list[str], why: str, landed: list[str],
                      trusted: set[str]) -> tuple[str, str, bool] | None:
        owners = self._owners(m)
        labels = {h: self.reviews(m.pr, h) for h in heads}
        verdicts = []
        for h, labs in labels.items():
            newest = {}
            for lab in sorted(labs, key=lambda lab: lab.ts):
                newest[_who(lab.by)] = lab
            mine = {_who(o) for o in (*owners, self.ready_by(m.pr, h)) if o}
            verdicts += [(lab.ts, lab.value == "hold", h, lab) for by, lab in newest.items()
                         if lab.value == "hold" or by not in mine]
        if not (lands := [v for v in verdicts if not v[1]]):
            return None
        old = max(lands, key=lambda v: v[0])[2]
        if why.startswith("held"):
            return old, f"{why} at {m.head[:12]}", False
        if (v := max(verdicts, key=lambda v: v[:2]))[1]:
            return old, f"{v[3].by.lstrip('@')}'s hold at {v[2][:12]} ({v[3].ref or 'no reason'}) is newer", False
        if held := unreviewed(labels[old], (*owners, self.ready_by(m.pr, old))):
            return old, f"{held} at {old[:12]}", False
        if _who(CI) in {_who(o) for o in owners if o}:
            return old, f"{CI} is among its owners, so its verdict wouldn't count", False
        d = self.delta(old, m.head, landed, trusted)
        return old, d.why if d.carries else f"{d.kind}: {d.why}", d.carries

    @staticmethod
    def _ran_on(states: dict) -> list[str]:
        return sorted({s[4] for s in states.values() if len(s) > 4 and s[4] and s[1] not in (NOT_STARTED, SIGTERM_RC)})

    def _elsewhere(self, states: dict, on: str = "") -> str:
        return elsewhere(on or self.on, self._ran_on(states))

    def _open(self) -> dict[int, PR]:
        if self._prs is None:
            self._prs = {p.number: p for p in self.q.github.open_prs()}
        return self._prs

    def tell(self, state: dict, keys: list[str], text: str, handle: str | None, *also: str | None) -> None:
        if keys and all(k in state["told"] for k in keys):
            return
        to = tuple(dict.fromkeys((f"@{handle}" if handle else CI, *(f"@{h}" for h in also if h))))
        if self.q.shadow or self.dry_run:
            self.q.log(f"research queue drive: would tell {' '.join(to)}: {text} [{'shadow' if self.q.shadow else 'dry run'}]")
            return
        if self.post(f"{' '.join(to)} queue: {text}", to):
            state["told"].extend(k for k in keys if k not in state["told"])
            self.q.log(f"research queue drive: told {' '.join(to)}: {text}")
        else:
            self.q.log(f"research queue drive: could not post to the merge thread (the next tick retries): {' '.join(to)}: {text}")

    def _stacked(self, state: dict, members: list[Member], tip: str) -> None:
        from .stale import pull_refs

        q = self.q
        landed = {x.pr for x in members}
        try:
            opened = self._open()
            rest = {n: p for n, p in opened.items() if n not in landed}
            refs = pull_refs(q, rest.values())
        except QueueError as e:
            q.log(f"research queue drive: PRs stacked on {', '.join(f'#{n}' for n in sorted(landed))} not found: {e}")
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
                q.log(f"research queue drive: #{n}'s ready mark can't be read, so it isn't told its base landed now: {e}")
                continue
            names = ", ".join(f"#{b}" for b in below)
            self.tell(state, [f"stacked:{n}<{b}" for b in below],
                      f"#{n}'s base{'s' if len(below) > 1 else ''} {names} landed in main at {tip[:12]}: "
                      f"{merge_main(f'#{n}', rest[n].branch)}",
                      self._handle(n))

    def beside_args(self, tip: str, base: str) -> tuple[dict, list[str], str]:
        from .merge import required_steps

        extra = {"agreement": "lean-agreement" in required_steps(self.repo, base, tip, "check"), "no_regenerate": True}
        skip = [] if extra["agreement"] else ["lean-agreement"]
        below = (base if base != tip else
                 _git(self.repo, "rev-parse", "--verify", "-q", f"{tip}^1", check=False).stdout.strip()) or None
        flags = (f" --base {below[:12]}" if below else "") + "".join(f" --{k.replace('_', '-')}" for k, on in extra.items() if on)
        return {"base": below, **extra}, skip, flags
