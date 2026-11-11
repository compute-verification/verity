"""`research queue stats` with no `--log`: the merge queue's core numbers as one table, read live in under two minutes.

Each row is one number with its unit, over three windows: now, the last hour, and the last 24 hours. What is read:

  the loop's decision log   `decisions.jsonl` beside the verified queue's loop on the control pod (`vqueue_loop`): each
                            candidate's build, its verdict as the loop judged it, its drop and why, and each move of `main`;
                            and the loop's own log for the time each tick ended
  the labels                `ready` and `reviewed` on `pr:N@HEAD` in the evidence store: a PR's mark and its land verdict
  the run directories       on every node, each run's `job.json` (its command and commit), `slot.json` (when it took a check
                            slot), `status.json` (when it ended), and for a check's shard `check-live.json`, `steps.jsonl` and
                            `suites.failed.json`; for a run of the task graph's executor, the result files under `out/results`
  the nodes                 `/proc/stat` twice, a few seconds apart, and `slot.py --status --json` for each slots file

A mark is the first `ready = true` label after the last `ready = false` at the head that landed (else the newest at any head
of the PR, a mark carried across a merge of `main`). A landing is the loop's `advance` of `main` to a candidate holding the PR.
A candidate's verdict is the loop's first `judged pass` or `judged fail` of the run beside its check. A red's first failure is
the earliest failed suite's result file, or failed step's end, among the candidate's shards.

`collect` does the reading and returns plain data; everything below it is a pure function of that data, so the table can be
rebuilt from a saved `--json` document. Stdlib only."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

CONTROL = "vy-control-verity"
#: the live loop's directory on the control pod: its decision log, its log, and the flags it runs with (`--on`: the check nodes)
LIVE = "/data/vqueue/live"
#: the CPU nodes: every one is read, since a shard in flight stays where it started when the loop's `--on` changes; the check
#: nodes are those of them the loop's `--on` names now
CPU_NODES = tuple(f"vy-nebius-cpu-{i}" for i in range(1, 14))
REPO = "compute-verification/verity"
#: the target: nine in ten PRs on `main` within two hours of the mark
TARGET_MIN, TARGET_SHARE = 120.0, 0.9
#: the steps off the merge gate when the checkout's policy.toml can't be read: a candidate's verdict doesn't wait for them
OFF_GATE = ("circuit-check", "ratchet-gate")
#: a file whose change is a Lean change: a module, a lakefile, a Lake manifest, a toolchain, a lock, or the Lean check's own
#: code less its tests and docs, most of which every package's audit is keyed on (`lean_audit.tool_files`)
LEAN_FILE = re.compile(r"\.lean$|(^|/)lakefile\.|(^|/)lake-manifest\.json$|(^|/)lean-toolchain$|(^|/)lean-audit\.json$"
                       r"|^tools/verity/lean/(?!tests/)(?!.*\.md$)|^tools/verity/check/common\.py$"
                       r"|^tools/verity/graph/verity_graph/contract\.py$")
#: the most files GitHub's compare lists: a candidate with this many and no Lean file among them is not known to change none
COMPARE_FILES = 300
#: the target for a round whose candidate changes no Lean, in minutes
PLAIN_ROUND_MIN = 30
#: a layout of shards (so many a candidate) has its rows while one of its shards runs or took its slot in the last so many hours
SHARD_LAYOUT_H = 3
#: the long window, in hours back from the read
WINDOW_H = 24
#: where the last reading of each node's CPU counters is kept, so the next run gives the average over the time between
STATE = Path.home() / ".research" / "queue-table.json"
#: this checkout, whose policy.toml and drive.toml say what is off the gate and what the low-risk rule admits
CHECKOUT = Path(__file__).resolve().parents[4]

CONTROL_SCRIPT = r'''
import ast, json, re, shlex, subprocess, sys
live, since = sys.argv[1], sys.argv[2]
rows = []
for line in open(live + "/decisions.jsonl", errors="replace"):
    if not line.startswith("{"):
        continue
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("at", "") < since or r.get("action") not in ("build", "advance", "drop", "slim", "compare", "retry", "eject"):
        continue
    keep = {k: r.get(k) for k in ("action", "at", "sha", "base", "members", "heads", "why", "shadow", "done", "cancel", "check",
                                  "graph2", "voided_why")}
    keep["why"] = (keep["why"] or "")[:400]
    keep["failed"] = [str((f or {}).get("task"))[:120] for f in (r.get("failed") or [])][:8]
    rec = r.get("record") or {}
    keep["ids"] = rec.get("ids") or []
    keep["verdict"] = (rec.get("verdict") or {}).get("state")
    rows.append(keep)
ticks = [m.group(1) for line in open(live + "/vqueue.log", errors="replace")
         if (m := re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) tick rc=", line)) and m.group(1) >= since]
try:
    args = open(live + "/args").read()
except OSError:
    args = ""
# the executor's own list of its checks: one line a check, its tasks by state (`graph2.py status`, which only reads)
checks, why = [], ""
try:
    words = shlex.split(args, comments=True)
    g2 = shlex.split(words[len(words) - words[::-1].index("--graph2")]) if "--graph2" in words[:-1] else []
    out = subprocess.run([*g2, "status"], capture_output=True, text=True, timeout=40).stdout if g2 else ""
    why = "" if out else "the loop's args name no --graph2, or its status printed nothing"
except (OSError, ValueError, subprocess.SubprocessError) as e:
    out, why = "", repr(e)[:200]
for line in out.splitlines():
    m = re.match(r"s(\d{8}T\d{6})Z-\w+ (\w+) \S+ (\{.*\})$", line)
    if m and m.group(1) >= since.replace("-", "").replace(":", ""):
        try:
            checks.append({"at": m.group(1), "sha": m.group(2), "tasks": ast.literal_eval(m.group(3))})
        except (ValueError, SyntaxError) as e:
            why = "a status line did not parse: " + repr(e)[:120]
print(json.dumps({"decisions": rows, "ticks": ticks, "args": args, "executor": checks, "executor_why": why}))
'''

NODE_SCRIPT = r'''
import calendar, glob, json, os, re, subprocess, sys, time
since, sample = sys.argv[1], float(sys.argv[2])
ROOT = "/workspace/research"

def ep(s):
    try:
        return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return None

def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

def mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None

def stat():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:9]]
    return [sum(v) - v[3] - v[4], sum(v)]

def cores(spec):
    try:
        return sum(int(b or a) - int(a) + 1 for a, _, b in (part.partition("-") for part in spec.split(",")))
    except (AttributeError, ValueError):
        return None

def used(path):
    """The run's CPU seconds and peak memory from its cgroup's last sample."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 65536))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        if '"cgroup"' not in line:
            continue
        try:
            cg = json.loads(line).get("cgroup") or {}
        except ValueError:
            continue
        return {"cpu_s": ((cg.get("cpu_stat") or {}).get("usage_usec") or 0) / 1e6,
                "mem_peak_gb": (cg.get("memory_peak_bytes") or 0) / 2 ** 30}
    return {}

def scope(rid):
    """A running check's own cgroup: its CPU seconds, its memory now without the page cache and its peak with it."""
    for d in glob.glob("/sys/fs/cgroup/user.slice/user-*.slice/user@*.service/app.slice/check-*-" + rid + ".scope"):
        try:
            cpu = dict(l.split() for l in open(d + "/cpu.stat") if len(l.split()) == 2)
            mem = dict(l.split() for l in open(d + "/memory.stat") if len(l.split()) == 2)
            row = {"scope_cpu_s": int(cpu.get("usage_usec", 0)) / 1e6, "scope_anon_gb": int(mem.get("anon", 0)) / 2 ** 30,
                   "scope_peak_gb": int(open(d + "/memory.peak").read()) / 2 ** 30}
        except (OSError, ValueError):
            return {}
        try:
            row["parallel"] = parallel(d)
        except OSError:
            row["parallel"] = None
        return row
    return {}

def status(pid):
    """A process's parent, its anonymous memory in GiB and its working directory: never its command line."""
    try:
        f = dict(l.split(":", 1) for l in open("/proc/%s/status" % pid) if ":" in l)
        return {"ppid": f["PPid"].strip(), "anon": int(f.get("RssAnon", "0").split()[0]) / 2 ** 20,
                "cwd": os.readlink("/proc/%s/cwd" % pid)}
    except (OSError, KeyError, ValueError, IndexError):
        return None

def declared(tree, suite):
    """What a suite's pyproject.toml declares in the checkout ``tree``: whether its tests run in parallel, a worker's GB, and
    the GB the runner takes for a worker that declares none; None where it is no suite."""
    try:
        text = open(tree + "/" + suite + "/pyproject.toml").read()
        runner = open(tree + "/tools/verity/check/suites.py").read()
    except OSError:
        return None
    sec = re.search(r"^\[tool\.verity\.tests\][ \t]*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    body = sec.group(1) if sec else ""
    gb = re.search(r"^worker_gb\s*=\s*([0-9.]+)", body, re.M)
    default = re.search(r"^TEST_PROCESS_GB\s*=\s*([0-9.]+)", runner, re.M)
    return {"parallel": bool(re.search(r"^parallel\s*=\s*true", body, re.M)), "worker_gb": float(gb.group(1)) if gb else None,
            "default_gb": float(default.group(1)) if default else None}

def parallel(d):
    """The parallel suites running in the cgroup ``d``, each with its workers' anonymous memory in GiB (``own``, RssAnon) and
    the same with the processes a worker started (``tree``, each page once: Pss_Anon). A suite's pytest is the process the
    runner started in the suite's directory; its children are its workers, and with fewer than two it runs the tests itself."""
    pids = []
    for base, _, files in os.walk(d):
        if "cgroup.procs" in files:
            pids += open(base + "/cgroup.procs").read().split()
    ps = {p: s for p in pids for s in [status(p)] if s}
    kids = {}
    for p, s in ps.items():
        kids.setdefault(s["ppid"], []).append(p)

    def shared(p):
        # a forked process maps its parent's pages, which RssAnon counts once for each; Pss_Anon counts each in proportion
        try:
            for l in open("/proc/%s/smaps_rollup" % p):
                if l.startswith("Pss_Anon:"):
                    return int(l.split()[1]) / 2 ** 20
        except (OSError, ValueError, IndexError):
            pass
        return ps[p]["anon"]

    def below(p):
        return [p] + [x for k in kids.get(p, []) for x in below(k)]

    def tree(p):
        return sum(map(shared, below(p))) if kids.get(p) else ps[p]["anon"]

    out = []
    for p, s in ps.items():
        top = ps.get(s["ppid"])
        if not top or not s["cwd"].startswith(top["cwd"] + "/"):
            continue
        suite = s["cwd"][len(top["cwd"]) + 1:]
        dec = declared(top["cwd"], suite)
        if not dec or not dec.pop("parallel"):
            continue
        ws = kids.get(p, []) if len(kids.get(p, [])) > 1 else [p]
        out.append(dict(dec, suite=suite, own=[round(ps[w]["anon"], 2) for w in ws], tree=[round(tree(w), 2) for w in ws]))
    return out

def slow(d):
    """A shard's three slowest suites that ran, and the slowest passed test among them, from the suites' reports."""
    rows = [r for name in ("suites.json", "suites-lean.json") for r in load(d + "/" + name) or []
            if isinstance(r, dict) and r.get("status") != "cached" and isinstance(r.get("seconds"), (int, float))]
    rows.sort(key=lambda r: -r["seconds"])
    test = None
    for r in rows[:3]:
        passed = (load((r.get("log") or "").rsplit(".", 1)[0] + ".json") or {}).get("passed")
        for t, sec in passed.items() if isinstance(passed, dict) else []:
            if isinstance(sec, (int, float)) and (test is None or sec > test[2]):
                test = [r.get("name"), t, sec]
    return {"suites": [[r.get("name"), r["seconds"], r.get("workers")] for r in rows[:3]], "slow_test": test}

s0, t0 = stat(), time.time()
runs = []
for d in sorted(glob.glob(ROOT + "/runs/r2*")):
    rid = os.path.basename(d)
    if rid < since:
        continue
    job = load(d + "/job.json") or {}
    argv = [str(a) for a in ((job.get("store") or {}).get("params") or {}).get("argv") or []]
    text = " ".join(argv)
    kind = ("shard" if "--shard" in argv and "check.py" in text else "worker" if "graph2.py" in text
            else "tier" if "--quick" in text and "suites.py" in text else "other")
    slot = load(d + "/slot.json") or {}
    tr = (load(d + "/status.json") or {}).get("transitions") or []
    last = tr[-1] if tr else {}
    row = {"id": rid, "kind": kind, "took": ep(slot.get("took")), "slot_state": slot.get("state"), "slot": slot.get("slot"),
           "state": last.get("state"), "end": ep(last.get("t_utc")) if "rc" in last else None, "rc": last.get("rc")}
    if kind != "other":
        row["slot_cores"] = cores(slot.get("cores"))
        row.update(used(d + "/resources.jsonl"))
    if kind == "shard":
        row["sha"] = (((job.get("run") or {}).get("remote") or {}).get("source_sha") or "")[:12]
        row["shard"] = argv[argv.index("--shard") + 1] if argv.index("--shard") + 1 < len(argv) else "?"
        if row["end"] is None:
            row.update(scope(rid))
        row.update(slow(d))
        steps = {}
        for s in (load(d + "/check-live.json") or {}).get("steps") or []:
            steps[s.get("step")] = {"state": s.get("state"), "seconds": s.get("seconds"), "start": ep(s.get("started_utc"))}
        try:
            for line in open(d + "/steps.jsonl"):
                rec = json.loads(line)
                steps.setdefault(rec.get("step"), {}).update(state=rec.get("status"), seconds=rec.get("seconds"),
                                                             end=ep(rec.get("ended_utc")))
        except (OSError, ValueError):
            pass
        row["steps"] = [dict(v, step=k) for k, v in steps.items()]
        failed = []
        for name in ("suites.failed.json", "suites-lean.failed.json"):
            for f in load(d + "/" + name) or []:
                if not isinstance(f, dict):
                    f = {"name": str(f)}
                base = (f.get("log") or "").rsplit(".", 1)[0]
                # the report's time; a suite killed at its time limit writes none, and its log's time is its last test's
                failed.append([f.get("suite") or f.get("name"), (base and mtime(d + "/suites/" + base + ".json")) or None])
        row["failed"] = failed
    elif kind == "worker":
        i = argv.index("-c") if "-c" in argv else len(argv)
        row["tasks"] = max(0, len(argv) - i - 3)
        res = sorted((m, p) for p in glob.glob(d + "/out/results/*.json") if (m := mtime(p)))
        row["results"] = [round(m) for m, _ in res]
        if res:
            own = (load(res[0][1]) or {}).get("seconds")       # the task's wall seconds: its `cpu_s` counts every core
            row["first_own_s"] = own if isinstance(own, (int, float)) else None
    runs.append(row)

slots = []
try:
    trees = sorted((e.path for e in os.scandir(ROOT + "/src") if os.path.exists(e.path + "/tools/verity/check/slot.py")),
                   key=os.path.getmtime)
except OSError:
    trees = []
for f in ([ROOT + "/locks/slots"] + sorted(glob.glob(ROOT + "/locks/*/slots"))) if trees else []:
    try:
        out = subprocess.run(["python3", trees[-1] + "/tools/verity/check/slot.py", "--status", "--json"], capture_output=True,
                             text=True, timeout=30, env=dict(os.environ, CHECK_SLOTS=f)).stdout
        doc = json.loads(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        continue
    ss = doc.get("slots") or []
    slots.append({"pool": "check" if f == ROOT + "/locks/slots" else f.split("/")[-2], "slots": len(ss),
                  "busy": sum(1 for s in ss if s.get("holder")), "blocked": sum(1 for s in ss if s.get("blocked") and not s.get("holder")),
                  "waiting": len(doc.get("waiters") or []), "mem": {s.get("name"): s.get("mem") for s in ss if s.get("mem")}})
time.sleep(max(0.0, sample - (time.time() - t0)))
s1 = stat()
print(json.dumps({"now": time.time(), "cores": os.cpu_count(), "load": open("/proc/loadavg").read().split()[:3],
                  "busy": (s1[0] - s0[0]) / max(1, s1[1] - s0[1]), "stat": s1, "hz": os.sysconf("SC_CLK_TCK"),
                  "slots": slots, "runs": runs}))
'''


# ---- reading ----------------------------------------------------------------------------------------------------------------

def _remote(machine: str, script: str, *args: str, timeout: float = 100) -> tuple[dict | None, str]:
    """``script`` under the machine's python3, its last JSON line parsed; here when this is that machine (the control pod)."""
    local = machine == CONTROL and Path(LIVE).is_dir()
    cmd = ["python3", "-", *args] if local else [sys.executable, "-m", "research", "pods", "ssh", machine, "--", "python3", "-", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")]))
    try:
        p = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, f"no answer in {timeout:g} s"
    answer = next((line for line in reversed(p.stdout.splitlines()) if line.startswith("{")), None)
    if answer is None:
        return None, (p.stderr or p.stdout).strip()[-200:] or f"rc {p.returncode}"
    try:
        return json.loads(answer), ""
    except ValueError as e:
        return None, f"its answer is not JSON: {e}"


def check_nodes(args: str) -> list[str]:
    """The nodes the loop sends a candidate's shards to: the last `--on` of its arguments file, as its parser reads them (a
    deploy appends a line, and a line starting with `#` is a comment)."""
    found = re.findall(r"(?:^|\s)--on[ =](\S+)", "\n".join(ln for ln in args.splitlines() if not ln.lstrip().startswith("#")))
    return found[-1].split(",") if found else []


def _labels(store_root, targets: list[str]) -> dict[str, list[dict]]:
    """Each target's `ready` and `reviewed` labels, pulled from the store's remote first."""
    from .store import labels_sync
    from .store.local import LocalStore

    store = LocalStore(store_root)
    try:
        labels_sync.pull_targets(store, targets, jobs=16)
    except Exception as e:  # noqa: BLE001  a VM without the store's secrets still reads what it pulled before
        print(f"research queue stats: labels not pulled ({e}); reading the local copies", file=sys.stderr)
    return {t: [{"key": lab.key, "value": lab.value, "by": lab.by, "ts": lab.ts} for lab in store.labels(t)
                if lab.key in ("ready", "reviewed")] for t in targets}


def lean_files(base: str, sha: str) -> list[int] | None:
    """[the files a candidate changes from its base, those of them that are Lean's], from GitHub's compare of the two commits, which
    lists at most `COMPARE_FILES`; None when it gave no answer."""
    try:
        p = subprocess.run(["gh", "api", f"repos/{REPO}/compare/{base}...{sha}", "--jq", "[.files[].filename]"],
                           capture_output=True, text=True, timeout=90)
        got = json.loads(p.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        print(f"research queue stats: no compare of {base[:12]}...{sha[:12]} ({e!r}); it counts as not known", file=sys.stderr)
        return None
    if not isinstance(got, list) or not all(isinstance(f, str) for f in got):
        return None
    return [len(got), sum(1 for f in got if LEAN_FILE.search(f))]


def pr_files(pr: str) -> list[str] | None:
    """The files an open PR changes from its base, as GitHub lists them (at most 3,000); None when it gave no answer."""
    try:
        p = subprocess.run(["gh", "api", "--paginate", f"repos/{REPO}/pulls/{pr}/files", "--jq", ".[].filename"],
                           capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"research queue stats: no files of #{pr} ({e!r}); it counts as needing a land verdict", file=sys.stderr)
        return None
    return p.stdout.splitlines() if p.returncode == 0 else None


def lowrisk_heads(open_prs: dict[str, str], labels: dict[str, list[dict]], now: float) -> dict[str, bool | None]:
    """{PR@head: whether the low-risk rule admits it on its ready mark alone}, for each open head marked ready that has no land
    verdict and no reviewer's hold: the loop takes such a head when its files match nothing the checkout's drive.toml excludes
    (`queue.lowrisk`).  None for a head GitHub listed no files for; nothing when the lane is off or drive.toml can't be read."""
    from .queue import DRIVE_FLAGS, DriveFlags, QueueError, lowrisk

    try:
        flags = DriveFlags.parse((CHECKOUT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS)
    except (OSError, QueueError) as e:
        print(f"research queue stats: {DRIVE_FLAGS} not read ({e}); no head counts as low-risk", file=sys.stderr)
        return {}
    if not (flags.lowrisk_lane and flags.lowrisk_excludes):
        return {}
    bare = [pr for pr, head in open_prs.items()
            if mark_at(labels.get(f"{pr}@{head}") or [], now) and verdict_at(labels.get(f"{pr}@{head}") or []) == (None, False)]
    with ThreadPoolExecutor(8) as ex:
        files = list(ex.map(pr_files, bare))
    return {f"{pr}@{open_prs[pr]}": None if got is None else lowrisk(got, flags.lowrisk_excludes) for pr, got in zip(bare, files)}


def collect(store_root=None, sample_s: float = 5.0, nodes: list[str] | None = None, now: float | None = None,
            lean: dict[str, list[int]] | None = None) -> dict:
    """Everything the table reads, as one JSON-able document.  ``lean`` is what earlier runs learned of `lean_files`, by a
    candidate's first 12 hex digits, so only a candidate not in it is asked for."""
    from .queue import label_target

    now = now or datetime.now(UTC).timestamp()
    y = datetime.fromtimestamp(now - WINDOW_H * 3600, UTC)
    day = f"{y:%Y-%m-%dT%H:%M:%SZ}"
    # the loop's lines from a day before the window: a PR that landed in the window was first built before it
    lead = datetime.fromtimestamp(now - (WINDOW_H + 24) * 3600, UTC)
    control, err = _remote(CONTROL, CONTROL_SCRIPT, LIVE, f"{lead:%Y-%m-%d}T00:00:00Z")
    doc: dict = {"now": now, "day": day, "errors": [f"{CONTROL}: {err}"] if control is None else [], "control": control or {}}
    checks = check_nodes((control or {}).get("args", ""))
    doc["check_nodes"] = checks
    names = nodes or [*checks, *(n for n in CPU_NODES if n not in checks)]
    with ThreadPoolExecutor(max(1, len(names))) as ex:
        got = list(ex.map(lambda n: _remote(n, NODE_SCRIPT, f"r{y:%Y%m%d}", str(sample_s)), names))
        gh = ex.submit(subprocess.run, ["gh", "pr", "list", "--repo", REPO, "--state", "open", "--limit", "2000", "--json",
                                        "number,headRefOid"], capture_output=True, text=True)
        doc["nodes"] = {n: d for n, (d, _) in zip(names, got) if d is not None}
        doc["errors"] += [f"{n}: {e}" for n, (d, e) in zip(names, got) if d is None]
        try:
            doc["open"] = {str(p["number"]): p["headRefOid"] for p in json.loads(gh.result().stdout)}
        except (ValueError, KeyError):
            doc["open"] = {}
            doc["errors"].append("gh pr list: no list of open PRs")
    try:
        import tomllib

        doc["off_gate"] = tomllib.loads((CHECKOUT / "tools/verity/check/policy.toml").read_text())["off_gate"]
    except (OSError, ValueError, KeyError, IndexError):
        doc["off_gate"] = list(OFF_GATE)
    built = {(str(pr), head) for r in doc["control"].get("decisions", []) if r["action"] in ("advance", "build") and not r.get("shadow")
             for pr, head in zip(r.get("members") or [], r.get("heads") or [])}
    builds = {r["sha"]: r["base"] for r in doc["control"].get("decisions", []) if r["action"] == "build" and not r.get("shadow")
              and r.get("sha") and r.get("base") and r["at"] >= day}
    known = dict(lean or {})
    ask = [sha for sha in builds if sha[:12] not in known]
    with ThreadPoolExecutor(8) as ex:
        known.update({sha[:12]: got for sha, got in zip(ask, ex.map(lambda sha: lean_files(builds[sha], sha), ask)) if got})
    doc["lean"] = {sha[:12]: known[sha[:12]] for sha in builds if sha[:12] in known}
    pairs = sorted(built | set(doc["open"].items()))
    labels = _labels(store_root, [label_target(int(pr), head) for pr, head in pairs])
    doc["labels"] = {f"{pr}@{head}": labels[label_target(int(pr), head)] for pr, head in pairs}
    doc["lowrisk"] = lowrisk_heads(doc["open"], doc["labels"], now)
    return doc


# ---- pure -------------------------------------------------------------------------------------------------------------------

def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def window_start(doc: dict) -> float:
    """When the long window starts: ``doc["day"]``, a UTC timestamp or a date (its 00:00Z)."""
    return ts(doc["day"] if "T" in doc["day"] else doc["day"] + "T00:00:00Z")


def rid_time(rid: str) -> float:
    """When a run's id was made: the moment its owner asked for the run."""
    return datetime.strptime(rid[1:16], "%Y%m%d-%H%M%S").replace(tzinfo=UTC).timestamp()


def pct(vals: list[float], q: float) -> float | None:
    """The q-quantile by linear interpolation between the two nearest values; None of nothing."""
    vals = sorted(vals)
    if not vals:
        return None
    k = q * (len(vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def mark_at(labels: list[dict], before: float) -> float | None:
    """The ready mark in force at ``before``: the first `ready = true` after the last `ready = false`, None when the newest is
    false or there is none."""
    mark = None
    for lab in sorted((x for x in labels if x["key"] == "ready"), key=lambda x: x["ts"]):
        at = ts(lab["ts"])
        if at > before:
            break
        mark = (mark or at) if str(lab["value"]).lower() == "true" else None
    return mark


def verdict_at(labels: list[dict]) -> tuple[float | None, bool]:
    """When the head first got a land verdict, and whether its newest verdict is a hold."""
    revs = sorted((x for x in labels if x["key"] == "reviewed"), key=lambda x: x["ts"])
    lands = [ts(x["ts"]) for x in revs if x["value"] == "land"]
    return (lands[0] if lands else None), bool(revs) and revs[-1]["value"] == "hold"


def drop_cause(why: str) -> str:
    """Why the loop dropped a candidate, from its reason: `on main`, `by hand`, `dropped base` (the candidate below it was
    dropped or failed), `withdrawn mark`, `infrastructure` (no run registered, a killed run), `hold` (a reviewer or lead held
    a member), `red` (a failed step, or a member ejected for one), else `other`."""
    if why.startswith("on main"):
        return "on main"
    if why.startswith("ci dropped it by hand"):
        return "by hand"
    if why.startswith(("its base", "not on main's chain")):
        return "dropped base"
    if re.match(r"#\d+ isn't taken now: held", why):
        return "hold"
    if re.match(r"#\d+ isn't taken now: ejected", why) or why.startswith("failed"):
        return "red"
    if "ready mark was withdrawn" in why:
        return "withdrawn mark"
    if "infra error" in why or "registered no run" in why:
        return "infrastructure"
    return "other"


def red_kind(why: str, failed: list[str]) -> str:
    """A red candidate's kind from the loop's reason and failed tasks: `test` where it names the pytest step or a test,
    whatever else its text mentions; else `infrastructure`, `preflight`, `Lean audit` (any Lean step), else `test`."""
    text = (why + " " + " ".join(failed)).lower()
    if "pytest" in text or re.search(r"\btests/\S+\.py::", text):
        return "test"
    if "infra" in text or "registered no run" in text:
        return "infrastructure"
    if "preflight" in text:
        return "preflight"
    return "Lean audit" if "lean" in text else "test"


def candidates(decisions: list[dict], runs: dict[str, dict], day_start: float, off_gate: tuple[str, ...] = OFF_GATE) -> list[dict]:
    """Each candidate built since ``day_start``: its build, verdict (the loop's first `judged pass` or `judged fail`), outcome
    (`green`: `main` moved to it; `red`: dropped for a failure, a member held over one, or infrastructure, with its kind;
    `dropped`: for another cause; `running`), its shards' runs, and for a red the first failure among them, the drop, and the
    next build after it. A failure is a suite's failed report or a failed step on the gate: a step in ``off_gate`` fails
    without making its candidate red."""
    out: dict[str, dict] = {}
    builds = sorted(ts(r["at"]) for r in decisions if r["action"] == "build" and not r.get("shadow"))
    for r in decisions:
        if r.get("shadow") or not r.get("sha"):
            continue
        at, sha = ts(r["at"]), r["sha"][:12]
        if r["action"] == "build" and at >= day_start:
            out[sha] = {"sha": sha, "build": at, "members": r.get("members") or [], "outcome": "running", "ids": []}
        c = out.get(sha)
        if c is None:
            continue
        if r["action"] == "slim":
            c["ids"] = r.get("ids") or c["ids"]
            if r.get("verdict") in ("pass", "fail") and "verdict" not in c:
                c["verdict"], c["verdict_state"] = at, r["verdict"]
        elif r["action"] == "advance" and r.get("done"):
            c["outcome"], c["landed"] = "green", at
        elif r["action"] == "drop" and c["outcome"] == "running":
            why = r.get("why") or ""
            c["cause"], c["dropped"], c["why"] = drop_cause(why), at, why[:160]
            red = c["cause"] in ("red", "infrastructure") or (c["cause"] == "hold" and ("red in" in why or "fail" in why))
            c["outcome"] = "red" if red else "dropped"
            if red:
                c["kind"] = red_kind(why, r.get("failed") or [])
                c["rebuilt"] = next((b for b in builds if b > at), None)
    for c in out.values():
        c["shards"] = sorted((runs[i] for i in c["ids"] if i in runs), key=lambda x: x.get("shard") or "")
        fails = [f[1] for s in c["shards"] for f in s.get("failed") or [] if f[1]]
        fails += [st["end"] for s in c["shards"] for st in s.get("steps") or []
                  if st.get("state") == "failed" and st.get("end") and st.get("step") not in off_gate]
        if c["outcome"] == "red" and fails:
            c["first_failure"] = min(fails)
    return sorted(out.values(), key=lambda c: c["build"])


def gate_end(shard: dict, off_gate: tuple[str, ...]) -> float | None:
    """When the last of a shard's steps on the merge gate ended, or None while one hasn't (or was stopped)."""
    steps = [st for st in shard.get("steps") or [] if st.get("step") not in off_gate]
    if not steps or any(not st.get("end") or st.get("state") == "stopped" for st in steps):
        return None
    return max(st["end"] for st in steps)


def parts(doc: dict) -> list[dict]:
    """Where each PR landed in the long window spent its time, in minutes: `review` (mark to land verdict), `to_build` (that to the first
    candidate holding it), `rework` (that build to the build of the candidate it landed in), `check` (that build to the
    candidate's verdict) and `to_main` (the verdict to the move of `main`)."""
    decisions, labels = doc["control"].get("decisions") or [], doc["labels"]
    day = window_start(doc)
    built: dict[tuple[str, str], list[float]] = {}
    build_at, judged = {}, {}
    for r in decisions:
        if r.get("shadow"):
            continue
        if r["action"] == "build":
            build_at[r["sha"]] = ts(r["at"])
            for pair in zip(map(str, r.get("members") or []), r.get("heads") or []):
                built.setdefault(pair, []).append(ts(r["at"]))
        elif r["action"] == "slim" and r.get("verdict") == "pass":
            judged.setdefault(r["sha"], ts(r["at"]))
    out = []
    for r in decisions:
        if r["action"] != "advance" or not r.get("done") or r.get("shadow") or ts(r["at"]) < day:
            continue
        landed = ts(r["at"])
        for pr, head in zip(map(str, r.get("members") or []), r.get("heads") or []):
            labs = labels.get(f"{pr}@{head}") or []
            mark = mark_at(labs, landed)
            if mark is None or r["sha"] not in build_at:
                continue
            v = max(mark, verdict_at(labs)[0] or mark)
            last = build_at[r["sha"]]
            first = min((b for b in built.get((pr, head), []) if b >= mark), default=last)
            ready = min(v, first)
            j = min(max(judged.get(r["sha"], landed), last), landed)
            out.append({"pr": pr, "landed": landed, "total": _m(landed - mark), "review": _m(ready - mark), "to_build": _m(first - ready),
                        "rework": _m(last - first), "check": _m(j - last), "to_main": _m(landed - j)})
    return out


def _m(seconds: float | None) -> float | None:
    return None if seconds is None else seconds / 60


def _dist(vals: list[float | None], unit: str = "min", digits: int = 0) -> str:
    """`p50 A, p90 B unit (n=N)` of the values that exist."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "none"
    return f"p50 {pct(vals, .5):.{digits}f}, p90 {pct(vals, .9):.{digits}f} {unit} (n={len(vals)})"


def earlier(readings: list[dict], now: float, want: float = 3600.0) -> dict | None:
    """Of the saved readings of the nodes' CPU counters, the one taken nearest ``want`` seconds before ``now``, among those 30
    minutes to 2 hours old; None without one."""
    ok = [r for r in readings if want / 2 <= now - r["now"] <= want * 2]
    return min(ok, key=lambda r: abs(now - r["now"] - want), default=None)


def worker_memory(shards: list[dict]) -> list[dict]:
    """The parallel suites running in ``shards``, one row for each suite and declaration (two candidates can declare
    differently): its workers and the shards they run in, a worker's anonymous memory in GiB (``p50``, ``most``; ``tree``
    is the most with the processes a worker started), and what a worker is declared to reach (``declared``; the runner's
    default where ``by_default``, None where neither was read)."""
    found: dict[tuple, dict] = {}
    for r in shards:
        for s in r.get("parallel") or []:
            gb = s.get("worker_gb") or s.get("default_gb")
            o = found.setdefault((s["suite"], gb, not s.get("worker_gb")), {"shards": 0, "own": [], "tree": []})
            o["shards"] += 1
            o["own"] += s["own"]
            o["tree"] += s["tree"]
    return [{"suite": suite, "declared": gb, "by_default": by_default, "workers": len(o["own"]), "shards": o["shards"],
             "p50": pct(o["own"], .5), "most": max(o["own"]), "tree": max(o["tree"])}
            for (suite, gb, by_default), o in sorted(found.items(), key=lambda kv: -max(kv[1]["own"], default=0)) if o["own"]]


def declares(m: dict) -> str:
    """What a suite's worker is declared to reach, and whether the most one held is more."""
    if m["declared"] is None:
        return "its declaration was not read"
    says = f"declares none, so the runner counts {m['declared']:g} GiB" if m["by_default"] else f"declares {m['declared']:g} GiB"
    return says + (f": **over, {m['most'] / m['declared']:.1f} times**" if m["most"] > m["declared"] else ": within it")


def table(doc: dict, prev: dict | None = None, readings: list[dict] | tuple = ()) -> list[dict]:
    """The rows: each `{row, name, now, hour, day, target, how}`; a cell that could not be read says what it needs.
    ``readings`` are this machine's saved ones, for the rows that only see what is running when the command reads."""
    now = doc["now"]
    hour, day = now - 3600, window_start(doc)
    hours = max(1e-9, (now - day) / 3600)
    decisions = doc["control"].get("decisions") or []
    labels, open_prs, lowrisk = doc["labels"], doc["open"], doc.get("lowrisk") or {}
    off_gate = tuple(doc.get("off_gate") or OFF_GATE)
    checks = [n for n in doc["check_nodes"] if n in doc["nodes"]]
    runs = {r["id"]: dict(r, node=n) for n, d in doc["nodes"].items() for r in d["runs"]
            if r.get("end") is None or rid_time(r["id"]) >= day}
    rows: list[dict] = []

    def add(row: str, name: str, now_: str = "", hour_: str = "", day_: str = "", target: str = "", how: str = "") -> None:
        rows.append({"row": row, "name": name, "now": now_, "hour": hour_, "day": day_, "target": target, "how": how})

    def win(xs: list[tuple[float, float | None]], since: float) -> list[float | None]:
        return [v for at, v in xs if at >= since]

    # 1. mark to main
    landings = [(ts(r["at"]), str(pr), head) for r in decisions if r["action"] == "advance" and r.get("done") and not r.get("shadow")
                for pr, head in zip(r.get("members") or [], r.get("heads") or [])]
    moves = [ts(r["at"]) for r in decisions if r["action"] == "advance" and r.get("done") and not r.get("shadow")]
    m2m = [(at, _m(at - mark) if (mark := mark_at(labels.get(f"{pr}@{head}") or [], at)) else None) for at, pr, head in landings]
    waiting = {pr: mark for pr, head in open_prs.items() if (mark := mark_at(labels.get(f"{pr}@{head}") or [], now))}
    waits = sorted(_m(now - m) for m in waiting.values())
    marks = [*waiting.values(), *(at - v * 60 for at, v in m2m if v is not None)]
    cands = candidates(decisions, runs, day, off_gate)
    flying = [c for c in cands if c["outcome"] == "running"]
    in_flight = {str(pr) for c in flying for pr in c["members"]}
    has_verdict = {pr for pr in waiting if verdict_at(labels.get(f"{pr}@{open_prs[pr]}") or [])[0]}

    def share(since: float) -> str:
        got = [v for v in win(m2m, since) if v is not None]
        late = sum(1 for m in waiting.values() if m >= since and now - m > TARGET_MIN * 60)
        if not got:
            return "none landed" + (f"; {late} marked in the window have waited over 2 h" if late else "")
        inside = sum(v <= TARGET_MIN for v in got)
        return f"{inside} of {len(got)} landed ({inside / len(got):.0%})" + (
            f"; counting the {late} marked in the window and waiting over 2 h as misses, {inside / (len(got) + late):.0%}" if late else "")

    how = "mark: the `ready` label at the head that landed; landing: the loop's `advance` of `main`; by landing time"
    add("1", "Mark to `main`", "", _dist(win(m2m, hour)), _dist(win(m2m, day)), "", how)
    add("1", "Share on `main` within 2 h of the mark", "", share(hour), share(day), f"{TARGET_SHARE:.0%}", "of PRs landed in the window")
    add("1", "Marks an hour", "", f"{sum(m >= hour for m in marks)}", f"{sum(m >= day for m in marks) / hours:.1f}",
        "", "marks in force on open PRs and on PRs landed; a mark withdrawn since is not counted")
    add("1", "Landings an hour (PRs; moves of `main`)", "", f"{len(win(m2m, hour))} PRs; {sum(m >= hour for m in moves)} moves",
        f"{len(win(m2m, day)) / hours:.1f} PRs an hour; {len(win(m2m, day))} PRs in {sum(m >= day for m in moves)} moves", "", "")
    add("1", "Marked and waiting now (the backlog)", (f"{len(waiting)} PRs: {len(has_verdict)} with a land verdict, {len(waiting) - len(has_verdict)} "
        f"without; {len(in_flight & set(waiting))} in a candidate being checked; oldest {waits[-1] / 60:.1f} h; {sum(w > TARGET_MIN for w in waits)} over 2 h")
        if waits else "0", "", "", "", "open PRs whose head has a mark in force")

    # 2. mark to land verdict
    review: list[tuple[float, float | None]] = []
    no_verdict = []
    for key, labs in labels.items():
        pr, head = key.split("@")
        mark = mark_at(labs, now)
        v, hold = verdict_at(labs)
        if mark and v:
            review.append((v, max(0.0, _m(v - mark))))
        elif mark and open_prs.get(pr) == head and not lowrisk.get(key):
            no_verdict.append((_m(now - mark), hold))
    unheld = [w for w, hold in no_verdict if not hold]
    admitted = sum(1 for low in lowrisk.values() if low)
    unknown = sum(1 for low in lowrisk.values() if low is None)
    add("2", "Mark to land verdict (the review wait)", "", _dist(win(review, hour)), _dist(win(review, day)), "",
        "first `reviewed = land` label at the head less its mark, 0 when the verdict came first; by verdict time")
    add("2", "Marked heads with no land verdict now", f"{len(no_verdict)}: {len(no_verdict) - len(unheld)} held by a reviewer, {len(unheld)} "
        "not looked at" + (f", the oldest of those marked {max(unheld):.0f} min ago" if unheld else "")
        + (f" ({unknown} of them with files GitHub did not list)" if unknown else "")
        + (f"; {admitted} more need none: low-risk, admitted on the mark alone" if admitted else ""), "", "", "oldest 15 min",
        "an open head marked ready with no `reviewed = land` label; left out when the low-risk rule admits it on its mark alone: "
        "not held, and no file its PR changes (GitHub) is one the checkout's drive.toml excludes (`lowrisk_excludes`)")

    # 3. the quick tier before a mark
    tiers = [r for r in runs.values() if r["kind"] == "tier"]
    wait = [(r["took"], _m(r["took"] - rid_time(r["id"]))) for r in tiers if r.get("took")]
    ran = [(r["end"], _m(r["end"] - r["took"])) for r in tiers if r.get("took") and r.get("end")]
    inline = [_m(now - rid_time(r["id"])) for r in tiers if not r.get("took") and not r.get("end") and r.get("slot_state") == "waiting"]
    add("3", "Quick tier: waiting for a slot", f"{len(inline)} in line" + (f", longest {max(inline):.0f} min so far" if inline else ""),
        _dist(win(wait, hour)), _dist(win(wait, day)), "", "runs whose command is `suites.py ... --quick`; the run id's time to its slot "
        "(`slot.json`'s `took`); by the time it took the slot")
    add("3", "Quick tier: running", f"{sum(1 for r in tiers if r.get('took') and not r.get('end'))} running",
        _dist(win(ran, hour)), _dist(win(ran, day)), "", "its slot to its end (`status.json`); by end time")
    by_node = []
    for n in sorted(doc["nodes"], key=lambda n: int(n.rsplit("-", 1)[1])):
        w = [_m(r["took"] - rid_time(r["id"])) for r in tiers if r["node"] == n and r.get("took")]
        q = sum(1 for r in tiers if r["node"] == n and not r.get("took") and not r.get("end") and r.get("slot_state") == "waiting")
        if w or q:
            by_node.append(f"cpu-{n.rsplit('-', 1)[1]}: " + (f"{pct(w, .5):.0f}/{pct(w, .9):.0f} (n={len(w)})" if w else "none started")
                           + (f", {q} in line" if q else ""))
    add("3", "Quick tier: wait by node, p50/p90 min", "", "", "; ".join(by_node) or "none", "", "the runs of the last 24 hours that took a slot")

    # 4. a candidate's check
    b2v = [(c["verdict"], _m(c["verdict"] - c["build"])) for c in cands if c.get("verdict")]
    b2l = [(c["landed"], _m(c["landed"] - c["build"])) for c in cands if c.get("landed")]

    def outcomes(since: float) -> str:
        cs = [c for c in cands if c["build"] >= since]
        n = {k: sum(c["outcome"] == k for c in cs) for k in ("green", "red", "dropped", "running")}
        return f"{len(cs)} built: {n['green']} green, {n['red']} red, {n['dropped']} dropped for another reason, {n['running']} being checked"

    add("4", "Candidate: build to verdict", f"{len(flying)} being checked" + (", built " + ", ".join(f"{_m(now - c['build']):.0f}" for c in flying)
        + " min ago" if flying else ""), _dist(win(b2v, hour)), _dist(win(b2v, day)), "15", "the loop's `build` to its first `judged pass` or "
        "`judged fail`; by verdict time")
    lean = doc.get("lean") or {}

    def plain(c: dict) -> bool | None:
        """Whether the candidate changes no Lean file from its base; None when that is not known."""
        got = lean.get(c["sha"][:12])
        return None if not got or (got[1] == 0 and got[0] >= COMPARE_FILES) else got[1] == 0

    def slowest(c: dict) -> tuple[dict | None, list | None, list | None]:
        """A round's slowest step on the gate, its slowest suite that ran and its slowest passed test, among its shards."""
        return (max((st for sh in c["shards"] for st in sh.get("steps") or [] if st.get("seconds") and st["step"] not in off_gate),
                    key=lambda st: st["seconds"], default=None),
                max((su for sh in c["shards"] for su in sh.get("suites") or [] if su[1]), key=lambda su: su[1], default=None),
                max((sh["slow_test"] for sh in c["shards"] if sh.get("slow_test")), key=lambda t: t[2], default=None))

    def named(what: str, found: list, key, seconds, show=str) -> str:
        """The ``what`` that was the slowest in the most rounds: its name, its p50 minutes there, and in how many rounds."""
        keys = [key(f) for f in found if f]
        if not keys:
            return ""
        top = max(sorted(set(keys)), key=keys.count)
        return (f"; slowest {what} {show(top)}, p50 {pct([seconds(f) / 60 for f in found if f and key(f) == top], .5):.0f} min, "
                f"the slowest in {keys.count(top)} of {len(found)}")

    def a_step(found: list) -> str:
        return named("step", found, lambda st: st["step"], lambda st: st["seconds"], lambda k: f"`{k}`")

    def a_suite(found: list) -> str:
        return named("suite", found, lambda su: su[0], lambda su: su[1], lambda k: f"`{k}`")

    def a_test(found: list) -> str:
        return named("test", found, lambda t: (t[0], t[1]), lambda t: t[2], lambda k: f"`{k[1]}` of suite `{k[0]}`")

    def plain_rounds(since: float) -> str:
        judged = [c for c in cands if c.get("verdict") and c["verdict"] >= since]
        rs = [c for c in judged if plain(c)]
        unknown = f" ({n} not known)" if (n := sum(plain(c) is None for c in judged)) else ""
        if not rs:
            return f"none of the {len(judged)} with a verdict{unknown}"
        steps, _, tests = zip(*map(slowest, rs))
        return _dist([_m(c["verdict"] - c["build"]) for c in rs]) + f", of {len(judged)} with a verdict{unknown}" + a_step(steps) + a_test(tests)

    def newest() -> str:
        """The newest candidate with a verdict, as one clause: the first candidate after a change of the loop is read here."""
        c = max((c for c in cands if c.get("verdict")), key=lambda c: c["verdict"], default=None)
        if c is None:
            return ""
        got, step = lean.get(c["sha"][:12]), slowest(c)[0]
        return (f"; the newest verdict: built {datetime.fromtimestamp(c['build'], UTC):%H:%MZ} in {len(c['shards'])} shards, "
                f"{_m(c['verdict'] - c['build']):.0f} min, " + (f"{got[1]} of its {got[0]} changed files a Lean change" if got else "its changed files not known")
                + (f", slowest step `{step['step']}` {step['seconds'] / 60:.0f} min" if step else ""))

    add("4", "Round with no Lean change: build to verdict", (f"{sum(1 for c in flying if plain(c))} of the {len(flying)} being checked change no Lean"
        if flying else "none being checked") + newest(), plain_rounds(hour), plain_rounds(day), str(PLAIN_ROUND_MIN),
        "the candidates whose files changed from their base hold no `.lean` file, lakefile, Lake manifest, toolchain or `lean-audit.json` "
        "and none of the Lean check's own code (`tools/verity/lean` less its tests and docs, which each package's audit is keyed on) "
        "(GitHub's compare); the slowest step on the gate among a round's shards (`steps.jsonl`) and the slowest passed test among each "
        "shard's three slowest suites (the suites' reports); by verdict time")
    add("4", "Candidate: build to `main`", "", _dist(win(b2l, hour)), _dist(win(b2l, day)), "", "green candidates: `build` to `advance`")
    add("4", "Candidates by outcome", "", outcomes(hour), outcomes(day), "", "by build time; red: dropped for a failure, a member held "
        "or ejected over one, or infrastructure")
    built = {c["sha"]: c["build"] for c in cands}
    unstarted = [(ts(r["at"]), _m(ts(r["at"]) - built[r["sha"][:12]])) for r in decisions
                 if r["action"] == "slim" and (r.get("why") or "").startswith("judged infra") and not r.get("ids")
                 and r.get("sha", "")[:12] in built]
    add("4", "Check that started no shard: build to its `infra` verdict", "", _dist(win(unstarted, hour)), _dist(win(unstarted, day)), "",
        "the loop's `judged infra` lines that name no shard's run (a failed plan or launch), each from its candidate's build; the "
        "loop then retries the candidate, so one candidate can count more than once")
    lately = {s["shard"].rsplit("/", 1)[-1] for c in cands for s in c["shards"] if s.get("shard") and s.get("took")
              and (not s.get("end") or s["took"] >= now - SHARD_LAYOUT_H * 3600)}
    for k in sorted({s.get("shard") for c in cands for s in c["shards"] if s.get("shard") and s["shard"].rsplit("/", 1)[-1] in lately}):
        ss = [s for c in cands for s in c["shards"] if s.get("shard") == k]
        whole = [(s, g) for s in ss if s.get("took") and (g := gate_end(s, off_gate))]
        slow = [max((st for st in s["steps"] if st.get("seconds") and st["step"] not in off_gate), key=lambda st: st["seconds"], default=None)
                for s, _ in whole]
        names = [st["step"] for st in slow if st]
        top = max(set(names), key=names.count) if names else None
        after = [_m(s["end"] - g) for s, g in whole if s.get("end")]
        add("4", f"Shard {k}: its slot to the end of its last step on the gate", f"{sum(1 for s in ss if s.get('took') and not s.get('end'))} running",
            _dist([_m(g - s["took"]) for s, g in whole if g >= hour]), _dist([_m(g - s["took"]) for s, g in whole])
            + (f"; slowest step `{top}` in {names.count(top)} of {len(names)}, p50 {pct([st['seconds'] / 60 for st in slow if st and st['step'] == top], .5):.0f} min" if top else "")
            + (f"; then it holds the slot p50 {pct(after, .5):.0f}, p90 {pct(after, .9):.0f} min more for steps off the gate" if after and pct(after, .9) >= 1 else ""),
            "", "shards whose every step on the gate ended (a cancelled shard is left out); `steps.jsonl`")

    # 5. a red, and the rounds lost
    reds = [c for c in cands if c["outcome"] == "red"]
    f2d = [(c["dropped"], _m(c["dropped"] - c["first_failure"])) for c in reds if c.get("first_failure")]
    d2b = [(c["rebuilt"], _m(c["rebuilt"] - c["dropped"])) for c in reds if c.get("rebuilt")]
    kinds = {k: sum(c.get("kind") == k for c in reds) for k in ("test", "Lean audit", "preflight", "infrastructure")}
    add("5", "Red: first failure to the candidate's drop", "", _dist(win(f2d, hour)), _dist(win(f2d, day)), "10",
        "the earliest failed suite's result file or failed step's end among the shards, to the loop's `drop`; "
        f"{len(reds) - len(f2d)} of the window's {len(reds)} reds recorded no failure time on a node")
    add("5", "Red: drop to the next candidate built", "", _dist(win(d2b, hour)), _dist(win(d2b, day)), "", "the loop's `drop` to its next `build`")
    add("5", "Reds by kind", "", "", ", ".join(f"{v} {k}" for k, v in kinds.items()), "", "from the loop's reason and failed tasks")
    lost: dict[str, int] = {}
    for c in cands:
        if c["outcome"] in ("red", "dropped"):
            lost[c["cause"]] = lost.get(c["cause"], 0) + len(c["members"])
    add("5", "PR-rounds lost, by cause", "", "", f"{sum(lost.values())}: " + ", ".join(f"{v} {k}" for k, v in sorted(lost.items(), key=lambda kv: -kv[1]))
        if lost else "0", "", "one PR in one candidate that was dropped before `main` moved to it, by the loop's reason for the drop")

    # 6. the executor
    workers = [r for r in runs.values() if r["kind"] == "worker"]
    done = sorted(m for r in workers for m in r.get("results") or [])
    over = [(m, r["results"][0] - rid_time(r["id"]) - r["first_own_s"]) for r in workers
            if (m := (r.get("results") or [None])[0]) and r.get("first_own_s") is not None]
    compares = {r["sha"]: r for r in decisions if r["action"] == "compare" and ts(r["at"]) >= day}
    cancelled = len({r["sha"] for r in decisions if r["action"] == "drop" and ts(r["at"]) >= day and r.get("cancel") == "cancelled"})
    wslots = pct([r.get("slot_cores") or 0 for r in workers if r.get("slot_cores")], .5)

    def working(since: float) -> str:
        rs = [r for r in workers if r.get("end") and r.get("took") and r.get("cpu_s") and r["end"] > r["took"] and r["end"] >= since]
        if not rs:
            return "none"
        return (f"{sum(r['cpu_s'] for r in rs) / sum(r['end'] - r['took'] for r in rs):.1f} on average, "
                f"p50 {pct([r['cpu_s'] / (r['end'] - r['took']) for r in rs], .5):.1f} (n={len(rs)} runs)")

    add("6", "Executor (in shadow, not the gate): tasks finished a minute", f"{sum(m >= now - 600 for m in done) / 10:.1f} (last 10 min)",
        f"{sum(m >= hour for m in done) / 60:.1f}", f"{sum(m >= day for m in done) / (hours * 60):.1f} ({len(done)} tasks in {len(workers)} runs)",
        "230", "result files under `out/results` of the executor's worker runs, by file time; 230 a minute is a 15-minute check")
    add("6", "Executor: cores working in a slot", "", working(hour), working(day), f"{wslots:.0f} (the slot)" if wslots else "",
        "worker runs that ended in the window: CPU seconds (the run's cgroup's last sample) over seconds in the slot")
    add("6", "Executor: overhead before a run's first task result", "", _dist(win(over, hour), "s"), _dist(win(over, day), "s"), "",
        "the run id's time to its first result file, less that task's own wall seconds; a run holds p50 "
        f"{pct([r.get('tasks') or 0 for r in workers], .5) or 0:.0f} tasks")
    submitted = [dict(c, at=rid_time("r" + c["at"][:8] + "-" + c["at"][9:])) for c in doc["control"].get("executor") or []]
    unfinished = [c["tasks"] for c in submitted if any(c["tasks"].get(k) for k in ("queued", "waiting", "leased"))]
    checked = {c["sha"][:12] for c in cands if c["outcome"] == "running"}
    stray = [c["tasks"] for c in submitted if c["sha"][:12] not in checked and any(c["tasks"].get(k) for k in ("queued", "waiting", "leased"))]

    def cached(since: float) -> str:
        cs = [c["tasks"] for c in submitted if c["at"] >= since and sum(c["tasks"].values())]
        if not cs:
            return "none"
        hit, every = sum(t.get("hit", 0) for t in cs), sum(sum(t.values()) for t in cs)
        return f"{hit / every:.0%} from the cache ({hit} of {every} tasks in {len(cs)} checks)"

    if submitted:
        add("6", "Executor: share served from the cache; tasks waiting on other tasks",
            f"{sum(t.get('waiting', 0) for t in unfinished)} waiting on other tasks, {sum(t.get('queued', 0) for t in unfinished)} waiting for a slot, "
            f"{sum(t.get('leased', 0) for t in unfinished)} given to a worker, in {len(unfinished)} checks"
            + (f"; {len(stray)} of those checks, with {sum(t.get(k, 0) for t in stray for k in ('queued', 'waiting', 'leased'))} of the tasks, "
               "are of no candidate being checked now" if stray else ""), cached(hour), cached(day), "",
            "the executor's own list of its checks (`graph2.py status`): a check's tasks by state; the share is tasks answered from the "
            "cache over all the tasks of the checks submitted in the window, so a task cancelled before it ran counts as not served")
    else:
        add("6", "Executor: share served from the cache; tasks waiting on other tasks",
            f"not measured: {doc['control'].get('executor_why') or 'the executor listed no check'}")
    add("6", "Executor: shadow checks", "", "", f"{sum(r.get('graph2') in ('pass', 'fail') for r in compares.values())} reached a pass or a fail, "
        f"{len(compares)} compared with the shards, {sum(r.get('check') == r.get('graph2') for r in compares.values())} agreed; "
        f"{cancelled} cancelled with their candidate", "", "the loop's `compare` lines, and its `drop` lines whose cancel of the executor's check went through")

    # 7. cores and slots
    def cores_row(name: str, names: list[str]) -> None:
        total = sum(doc["nodes"][n]["cores"] for n in names)
        busy = sum(doc["nodes"][n]["busy"] * doc["nodes"][n]["cores"] for n in names)
        span = "not measured: needs a run of this command on this machine 30 min to 2 h ago (it saves each node's counters)"
        if prev and names and all(n in prev.get("nodes", {}) for n in names):
            dt = now - prev["now"]
            used = sum((doc["nodes"][n]["stat"][0] - prev["nodes"][n]["stat"][0]) / doc["nodes"][n]["hz"] for n in names)
            if used >= 0:
                span = f"{used / dt:.0f} of {total} ({used / dt / total:.0%}), the average over the last {dt / 60:.0f} min"
        add("7", name, f"{busy:.0f} of {total} ({busy / max(1, total):.0%})", span, "", "",
            "now: the non-idle share of `/proc/stat` over a few seconds on each node, times its cores; the hour: each node's busy CPU "
            f"seconds between two runs of this command; {len(names)} nodes read")

    cores_row("Cores busy over cores paid, the check nodes", checks)
    cores_row("Cores busy over cores paid, every CPU node", sorted(doc["nodes"]))
    pools: dict[str, list[int]] = {}
    for n in checks:
        for s in doc["nodes"][n]["slots"]:
            p = pools.setdefault(s["pool"] if s["pool"] in ("check", "executor") else "lane", [0, 0, 0, 0])
            for i, k in enumerate(("slots", "busy", "blocked", "waiting")):
                p[i] += s[k]
    add("7", "Slots busy over slots, the check nodes", "; ".join(f"{k} slots: {v[1]} of {v[0]} busy, {v[2]} free but blocked, {v[3]} runs in line"
                                                                for k, v in sorted(pools.items())) or "not read", "", "", "",
        "`slot.py --status --json` for each slots file; the lane pools share cores with each other and with the check slots, so slots "
        "add up to more than the cores")
    mem = {(n, name_): gb for n, d in doc["nodes"].items() for s in d["slots"] for name_, gb in (s.get("mem") or {}).items()}
    going = [dict(r, cores=r["scope_cpu_s"] / (doc["nodes"][r["node"]]["now"] - r["took"])) for r in runs.values()
             if r["kind"] == "shard" and r.get("scope_cpu_s") and r.get("took") and doc["nodes"][r["node"]]["now"] > r["took"]]
    res = [mem[(r["node"], r["slot"])] for r in going if (r["node"], r.get("slot")) in mem]
    live = sum(1 for r in runs.values() if r["kind"] == "shard" and r.get("took") and r.get("end") is None)
    add("7", "Reserved against used, a check's shard", (f"{len(going)} of the {live} running, the ones whose cgroup was found: cores p50 {pct([r['cores'] for r in going], .5):.0f}, most "
        f"{max(r['cores'] for r in going):.0f}, of {pct([r.get('slot_cores') or 0 for r in going], .5):.0f} in its slot; memory without the page "
        f"cache p50 {pct([r['scope_anon_gb'] for r in going], .5):.0f} GiB, most {max(r['scope_anon_gb'] for r in going):.0f} GiB; peak with the "
        f"page cache most {max(r['scope_peak_gb'] for r in going):.0f} GiB" + (f", of {pct(res, .5):.0f} GiB reserved" if res else ""))
        if going else f"no cgroup was found for the {live} running" if live else "no shard is running", "",
        "not measured: a shard's cgroup is gone when it ends", "",
        "the shards running now, from each one's own cgroup (`check-SLOT-RUN.scope`): CPU seconds over its seconds in the slot so far, "
        "`anon` of `memory.stat`, and `memory.peak`, which counts the page cache")
    read = [r for r in runs.values() if r["kind"] == "shard" and r.get("parallel") is not None]
    held = worker_memory(read)
    seen = [*(r for r in readings if "workers" in r), {"now": now, "workers": held}]

    def workers_seen(since: float) -> str:
        rs = [r for r in seen if r["now"] >= since]
        most: dict[tuple, dict] = {}
        for r in rs:
            for m in r["workers"]:
                o = most.setdefault((m["suite"], m["declared"], m["by_default"]), dict(m, readings=0))
                o["readings"] += 1
                o["most"] = max(o["most"], m["most"])
        return "; ".join(f"`{m['suite']}`: most {m['most']:.1f} GiB a worker, running at {m['readings']} of {len(rs)} readings; {declares(m)}"
                         for m in sorted(most.values(), key=lambda m: -m["most"])) or f"no parallel suite was running in a shard at any of {len(rs)} readings"

    add("7", "Parallel suites running in a shard: a worker's memory against what it declares", "; ".join(
        f"`{m['suite']}`: {m['workers']} workers in {m['shards']} shard{'s' * (m['shards'] != 1)}, a worker holds p50 {m['p50']:.1f}, most {m['most']:.1f} GiB"
        + (f" ({m['tree']:.1f} GiB with the processes it started)" if m["tree"] > m["most"] + 1 else "") + f"; {declares(m)}" for m in held)
        or (f"no parallel suite is running in the {len(read)} shard{'s' * (len(read) != 1)} running" if read else "no shard is running"),
        workers_seen(hour), workers_seen(day), "",
        "only the parallel suites running in a shard when this command read it: a suite that ran between two readings is not "
        "seen. Each process of a running shard's cgroup, grouped by the working directory of the pytest the runner started for "
        "a suite; a worker is a child of that pytest, its memory `RssAnon` of `/proc/PID/status` (no page cache, no shared "
        "memory); with the processes it started, each page once (`Pss_Anon` of `smaps_rollup`); the declaration is `worker_gb` of `[tool.verity.tests]` in the suite's `pyproject.toml` at the candidate's "
        "commit, else `TEST_PROCESS_GB` of `suites.py`, which the runner divides the slot's GiB by; the hour and the 24 hours: the same "
        "over this machine's saved readings")
    for kind, name in (("tier", "a quick tier"), ("worker", "an executor run")):
        rs = [r for r in runs.values() if r["kind"] == kind and r.get("end") and r.get("took") and r.get("cpu_s") and r["end"] > r["took"]]
        res = [mem[(r["node"], r["slot"])] for r in rs if (r["node"], r.get("slot")) in mem]
        add("7", f"Reserved against used, {name}", "", "", (f"cores: p50 {pct([r['cpu_s'] / (r['end'] - r['took']) for r in rs], .5):.1f} used of "
            f"{pct([r.get('slot_cores') or 0 for r in rs], .5):.0f} in its slot; memory: p50 {pct([r.get('mem_peak_gb') or 0 for r in rs], .5):.0f} GiB peak"
            + (f" of {pct(res, .5):.0f} GiB reserved" if res else ", its slot reserves none") + f" (n={len(rs)})") if rs else "none ended", "",
            "the window's runs that ended: CPU seconds and peak memory from the run's cgroup's last sample; the slot's cores and `mem=`")

    # 8. what else explains waiting
    ps = parts(doc)
    for key, name, how in (("to_build", "Ready to built", "the later of a PR's mark and land verdict to the first candidate built with it"),
                           ("rework", "First candidate to the one it landed in", "the build of the first candidate holding the PR to "
                            "the build of the candidate `main` moved to: the rounds lost")):
        add("8", name, "", _dist([p[key] for p in ps if p["landed"] >= hour]), _dist([p[key] for p in ps]), "", how + "; PRs landed in the window")
    ticks = sorted(ts(t) for t in doc["control"].get("ticks") or [])
    gaps = [(b, _m(b - a)) for a, b in zip(ticks, ticks[1:])]
    v2l = [(c["landed"], _m(c["landed"] - c["verdict"])) for c in cands if c.get("landed") and c.get("verdict")]
    b2s = [(max(s["took"] for s in c["shards"]), _m(max(s["took"] for s in c["shards"]) - c["build"]))
           for c in cands if c["shards"] and all(s.get("took") for s in c["shards"])]
    add("8", "The loop's tick: minutes between tick ends", f"last ended {_m(now - ticks[-1]):.0f} min ago" if ticks else "not read",
        _dist(win(gaps, hour), digits=1), _dist(win(gaps, day), digits=1), "", "the loop's log; every build, verdict, drop and move of `main` waits for a tick")
    add("8", "Candidate: build to its last shard holding a slot", "", _dist(win(b2s, hour)), _dist(win(b2s, day)), "",
        "the loop's `build` to the latest `took` among its shards: the launch and the wait for a check slot")
    add("8", "Green candidate: verdict to `main`", "", _dist(win(v2l, hour)), _dist(win(v2l, day)), "",
        "`judged pass` to `advance`: the wait for the candidates below it and for the tick")
    def floor(since: float) -> str:
        rs = [c for c in cands if c.get("verdict") and c["verdict"] >= since]
        _, suites, tests = zip(*map(slowest, rs)) if rs else ((), (), ())
        return (a_suite(suites) + a_test(tests))[2:] or "none"

    add("8", "The floor: a round's slowest suite and slowest test", "", floor(hour), floor(day), "",
        "what a round's tests take with unlimited cores: the slowest suite that ran among its shards (`suites.json`), or its slowest "
        "passed test where that suite is split (the suites' reports, of each shard's three slowest suites); rounds with a verdict in "
        "the window; the longest chain of Lean modules is not read")
    return rows


def largest(doc: dict) -> str:
    """The part of mark-to-`main` that took the most PR-minutes among the PRs landed in the long window, as one sentence."""
    ps = parts(doc)
    if not ps:
        return f"No PR landed in the last {WINDOW_H} hours, so no part of mark-to-`main` can be ranked."
    names = {"review": "the wait for a land verdict", "to_build": "ready to built: the wait for a first candidate",
             "rework": "rounds lost: the first candidate to the one that landed", "check": "the landing candidate's check",
             "to_main": "the candidate's verdict to the move of `main`"}
    total = {k: sum(p[k] for p in ps) for k in names}
    k = max(total, key=total.get)
    return (f"Largest part of mark-to-`main` for the {len(ps)} PRs landed in the last {WINDOW_H} hours: {names[k]}, p50 {pct([p[k] for p in ps], .5):.0f} min, "
            f"{total[k] / sum(total.values()):.0%} of all their minutes (" + ", ".join(
                f"{names[x].split(':')[0]} {total[x] / sum(total.values()):.0%}" for x in sorted(names, key=lambda x: -total[x]) if x != k) + ").")


def render(rows: list[dict], doc: dict, short: bool = False) -> str:
    """The table as Markdown; ``short`` leaves out the column saying how each number is measured."""
    at = datetime.fromtimestamp(doc["now"], UTC)
    cols = {"row": "#", "name": "Number", "now": "Now", "hour": "Last hour", "day": f"Last {WINDOW_H} h", "target": "Target", "how": "How it is measured"}
    if short:
        del cols["how"]
    lines = [f"The merge queue's numbers at {at:%Y-%m-%d %H:%MZ}. Windows: now; the last hour; the last {WINDOW_H} hours. Times in minutes unless said.",
             "", "| " + " | ".join(cols.values()) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[k]).replace("|", "/") for k in cols) + " |" for r in rows]
    lines += ["", largest(doc)]
    if doc["errors"]:
        lines += ["", "Not read: " + "; ".join(doc["errors"])]
    return "\n".join(lines)


def saved(path: Path) -> tuple[list[dict], dict[str, list[int]]]:
    """What earlier runs saved in ``path``: the readings (the nodes' CPU counters, and the parallel suites' workers seen running),
    and each candidate's `lean_files`; nothing without the file, or with one that doesn't parse (said on stderr)."""
    if not path.is_file():
        return [], {}
    try:
        doc = json.loads(path.read_text())
        # a count made under another rule for what a Lean file is says nothing under this one
        lean = dict(doc.get("lean") or {}) if doc.get("lean_rule") == LEAN_FILE.pattern else {}
        return [r for r in doc["readings"] if "now" in r and "nodes" in r], lean
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"research queue stats: {path} is unreadable ({e!r}); starting it again", file=sys.stderr)
        return [], {}


def main(store_root=None, as_json: bool = False, sample_s: float = 5.0, short: bool = False) -> int:
    readings, lean = saved(STATE)
    doc = collect(store_root, sample_s, lean=lean)
    rows = table(doc, earlier(readings, doc["now"]), readings)
    readings = [r for r in readings if doc["now"] - r["now"] < 26 * 3600]
    readings.append({"now": doc["now"], "nodes": {n: {"stat": d["stat"]} for n, d in doc["nodes"].items()},
                     "workers": worker_memory([r for d in doc["nodes"].values() for r in d["runs"]
                                               if r["kind"] == "shard" and r.get("parallel") is not None])})
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"readings": readings, "lean": {**lean, **doc["lean"]}, "lean_rule": LEAN_FILE.pattern}))
    except OSError as e:
        print(f"research queue stats: {STATE} not written ({e}): the next run has no reading from this one for its last-hour cores",
              file=sys.stderr)
    print(json.dumps({"at": doc["now"], "rows": rows, "errors": doc["errors"], "data": doc}, default=str) if as_json else render(rows, doc, short))
    return 0


if __name__ == "__main__":
    sys.exit(main(as_json="--json" in sys.argv[1:], short="--short" in sys.argv[1:]))
