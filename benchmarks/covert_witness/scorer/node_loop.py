"""The agent VM's side of the scorer's refresh, which runs on a CPU node (refresh-node.sh; the agent VM only edits): it
launches the runs, reads their small results, and does only what a node can't, which holds no console token, no store
write key and no push to origin.

    python node_loop.py init         # NODE's repository (K/repo.git), the VM's refs pushed, and the bootstrap run
    python node_loop.py once         # one refresh now: `push`, the served panels read, the run launched and `finish`ed
    python node_loop.py finish RUN   # a refresh run's out/vm/: its console bodies PUT, its board lines posted
    python node_loop.py loop         # `once` at each SLOT_S slot from FIRST until END

For each refresh run the VM pushes into the node's repository origin's branches, this VM's cursor/te-* and
covert-witness branches and every commit the scorer's records name that it holds (`push`); GETs the panels the route
serves for training (served.json, the rows the run's algorithm-policy keeps); launches refresh-node.sh with them and
the last refresh's PUT statuses (vm-put.log); and once it has ended and custody has stored its record, fetches
out/vm/'s small files, PUTs its console bodies, frontier-fit then algorithm-policy as te_core.algorithm_policy does,
where the route still serves what the run read, posts post.json's lines with those statuses in place of
v1cycle.VM_PUT, pushes merge.bundle to RESCORE_AHEAD, copies the panels and the efficiency table to the artifacts and
labels the run's record, so the store keeps it past a run record's 7 days.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
sys.path.insert(0, str(WT / "tools" / "research" / "src"))

from research import remote  # noqa: E402

import scorer as S  # noqa: E402
import v1cycle as V  # noqa: E402

NODE = os.environ.get("TE_NODE", "vy-nebius-cpu-7")
#: the refresh's state on the node (refresh-node.sh's K) and the research repository whose objects its first push borrows
K = os.environ.get("TE_NODE_STATE", "/workspace/research/lanes/training/te-scorer")
BORROW = "/workspace/research/git/verity.git/objects"
#: this loop's own small files: each run's served.json, the last refresh's PUT statuses, the commits to push, its log
LOOP = Path(os.environ.get("TE_NODE_LOOP", "/tmp/scorer/node"))
#: the VM's checkout of the re-score's te_core, whose console GETs and PUTs (the token is this VM's)
RESCORE = Path(os.environ.get("TE_RESCORE_WT", "/tmp/scorer/rescore"))
FIRST = datetime(2026, 11, 2, 21, 30, tzinfo=UTC).timestamp()
SLOT_S = 5400
END = datetime(2026, 11, 4, 16, 0, tzinfo=UTC).timestamp()
#: a slot missed by less than this starts at once
LATE_S = 600
#: how long a refresh's shards trace beside it, from its slot (refresh-node.sh, v1cycle's TE_HERE_UNTIL), and its run's cap
UNTIL_S = 80 * 60
TIMEOUT = "88m"
#: how long a run waits for the pool's slot before `research run` gives up on it
POOL_WAIT_S = 1200
MEM_GB = 200
#: a refresh run's files the VM reads (out/vm/), small: no trace and no table travels here
FILES = tuple(f"out/vm/{n}" for n in (
    "cycle.json", "digest.txt", "post.json", "stamp", "collect.json", "collect.out", "score.err", "v1cycle.log",
    "du.txt", "bootstrap-du.txt", "merge.bundle", "te-night2-efficiency.md", "put/algorithm-policy.json",
    "put/frontier-fit.json", f"panels/{V.FB}.json", f"panels/{V.FB}.png"))
ARTIFACTS = Path("/opt/cursor/artifacts")
CONSOLE_PY = Path(os.environ.get("TE_CONSOLE_PYTHON", "/workspace/.venv/bin/python"))
SERVED = "import json; from te_core import console; print(json.dumps(console.served()))"
PUT = ("import json, sys\nfrom te_core import console\nfor f in sys.argv[1:]:\n    p = json.load(open(f))\n"
       "    print(f\"PUT {console.PANELS_URL}/{p['id']}: {console.put(p)}\", flush=True)\n")


def say(*a) -> None:
    line = f"{datetime.now(UTC).strftime('%H:%M:%SZ')} " + " ".join(str(x) for x in a)
    print(line, flush=True)
    LOOP.mkdir(parents=True, exist_ok=True)
    with (LOOP / "loop.log").open("a") as f:
        f.write(line + "\n")


def ssh() -> list[str]:
    t = remote.ssh_host_target(remote.resolve_machine(NODE, "verity"))
    return remote.ssh_argv(t["host"], t["port"], user=t["user"])


def on_node(cmd: str, timeout: float = 120) -> str:
    r = remote.ssh_run(ssh(), ["sh", "-c", cmd], timeout=timeout)
    if r.returncode:
        raise SystemExit(f"node_loop: on {NODE}: {cmd[:120]}: rc {r.returncode}: {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout.decode(errors="replace")


def console(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(CONSOLE_PY), "-c", code, *args], cwd=RESCORE / "benchmarks" / "covert_witness",
                          capture_output=True, text=True, check=False,
                          env=os.environ | {"PYTHONPATH": str(WT / "tools" / "research" / "src")})


def git(*args: str, cwd: Path = WT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False)


def commits() -> list[str]:
    """The commits the scorer's records and runs name that this VM's repository holds: the VM's last index and runs,
    and the commits the node's last collect found in neither its repository nor origin (collect.json's commits_missing)."""
    want = set()
    for line in (S.STATE / "index.jsonl").open():
        x = json.loads(line)
        want |= {x.get(k) for k in ("commit", "measurement_commit")}
    want |= {a.get("commit") for a in json.loads((S.STATE / "runs.json").read_text())}
    if (LOOP / "missing.json").exists():
        want |= set(json.loads((LOOP / "missing.json").read_text()))
    want = sorted(c for c in want if isinstance(c, str) and re.fullmatch(r"[0-9a-f]{40}", c))
    r = subprocess.run(["git", "-C", str(WT), "cat-file", "--batch-check=%(objectname) %(objecttype)"],
                       input="\n".join(want) + "\n", capture_output=True, text=True, check=True)
    return [x.split()[0] for x in r.stdout.splitlines() if x.endswith(" commit")]


def push() -> str:
    """origin fetched here, then origin's branches, this VM's te-* and covert-witness branches and `commits` pushed
    into the node's repository (its own, so forced: it mirrors this VM's view)."""
    git("fetch", "-q", "origin")
    url = remote.git_push_url(ssh(), f"{K}/repo.git")
    refs = ["+refs/remotes/origin/*:refs/remotes/origin/*", "+refs/heads/cursor/te-*:refs/heads/cursor/te-*",
            "+refs/heads/cursor/covert-witness*:refs/heads/cursor/covert-witness*",
            *(f"+{c}:refs/te-scorer/c/{c}" for c in commits())]
    r = subprocess.run(["git", "-C", str(WT), "-c", "protocol.ext.allow=always", "push", "-q", "--no-verify", url, *refs],
                       capture_output=True, text=True, check=False)
    if r.returncode:
        raise SystemExit(f"node_loop: push to {NODE}:{K}/repo.git: rc {r.returncode}: {r.stderr[-600:]}")
    return f"pushed {len(refs) - 3} commits and the branches"


def launch(env: dict[str, str], send: list[Path], mem_gb: float, timeout: str) -> str:
    cmd = ["research", "run", "--on", NODE, "--pool", "research", "--as", V.TRACE_LANE, "--project", "verity",
           "--source", str(WT), "--cwd", "source", "--store-read", "--mem-gb", f"{mem_gb:g}", "--timeout", timeout,
           "--pool-wait", str(POOL_WAIT_S),
           "--custody-ttl", "8h", *(x for f in send for x in ("--send", str(f))),
           *(x for k, v in env.items() for x in ("--env", f"{k}={v}")),
           "--", "bash", "benchmarks/covert_witness/scorer/refresh-node.sh"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False, env=os.environ | {"AGENT_GUARD_LOCAL_OK": "1"})
    said = r.stdout + r.stderr
    run = next((w for w in said.split() if w.startswith("r20") and w.count("-") == 2), None)
    if r.returncode or not run or " refused" in said:
        raise SystemExit(f"node_loop: launch refused (rc {r.returncode}): {said.strip()[-600:]}")
    say("launched", run, "on", NODE)
    return run


def state(run: str) -> str:
    """The run's state as its fetched records say (remote.fetch), with out/vm/'s files beside them."""
    d = Path.home() / ".research" / "runs" / run
    rec = remote.fetch(d, also=FILES, timeout=300)
    return ((rec.get("last_observed") or {}).get("state")) or "unknown"


def wait(run: str, until: float) -> str:
    while True:
        st = state(run)
        if st in V.ENDED or time.time() > until:
            return st
        time.sleep(60)


def record(run: str, until: float) -> str | None:
    """The run's record (its attempt's run_record), once custody has stored it, waited for until `until`."""
    while True:
        a = V.attempt(run)
        art = (a.get("outputs") or {}).get("run_record")
        if art or time.time() > until:
            return art
        time.sleep(30)


def init() -> int:
    on_node(f"mkdir -p {K} && {{ [ -d {K}/repo.git ] || {{ git init -q --bare {K}/repo.git && "
            f"echo {BORROW} > {K}/repo.git/objects/info/alternates; }}; }}")
    say(push())
    LOOP.mkdir(parents=True, exist_ok=True)
    tar = LOOP / "state.tar"
    v1, gates = Path("/tmp/scorer/v1"), Path("/tmp/scorer/v1/2026-11-04T033103Z/algorithm-policy/analysis-gates.json")
    seed = LOOP / "seed" / "v1" / "trial"
    seed.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gates, seed / "gates.json")
    home = Path.home() / ".research"
    subprocess.run(["tar", "-cf", str(tar), "--exclude=./te-scorer/records",
                    "--transform=s,^\\./te-scorer,state,;s,^\\./te-ledger,ledger,", "-C", str(home),
                    "./te-scorer", "./te-ledger", "-C", str(v1.parent), "v1", "-C", str(LOOP / "seed"), "v1/trial"],
                   check=True)
    wts = []
    layers = sorted(p.name for p in Path("/tmp/scorer").glob("te-next*") if (p / ".git").exists())
    for name in ("te40", "rescore", "te-core", "te-core-pin", *layers):
        d = Path("/tmp/scorer") / name
        files = any(p.name != ".git" for p in d.iterdir())
        wts.append(f"{name}:{git('rev-parse', 'HEAD', cwd=d).stdout.strip()}:{'files' if files else 'nofiles'}")
    who = {k: git("config", k).stdout.strip() for k in ("user.name", "user.email")}
    run = launch({"MODE": "bootstrap", "TE_WORKTREES": ",".join(wts), "TE_GIT_NAME": who["user.name"],
                  "TE_GIT_EMAIL": who["user.email"]}, [tar], 16, "60m")
    st = wait(run, time.time() + 3600)
    say("bootstrap", run, st, (Path.home() / ".research" / "runs" / run / "out" / "vm" / "bootstrap-du.txt").exists())
    tar.unlink()
    return 0 if st == "done" else 1


def fix(text: str, put: dict[str, str]) -> str:
    for k, s in put.items():
        text = text.replace(f"training/{k} {V.VM_PUT}", f"training/{k} {'PUT ' if s[:1].isdigit() else ''}{s}")
    return text


def finish(run: str, until: float) -> dict:
    """Run `run`'s small results acted on (module docstring), once it ends (waited for until `until`); what it did."""
    d = Path.home() / ".research" / "runs" / run
    vm, mine = d / "out" / "vm", LOOP / run
    mine.mkdir(parents=True, exist_ok=True)
    st = wait(run, until)
    art = record(run, time.time() + 900)
    cycle = json.loads((vm / "cycle.json").read_text()) if (vm / "cycle.json").exists() else {}
    ap = cycle.get("algorithm_policy") or {}
    put = {}
    if ap.get("algorithm-policy") == V.VM_PUT and ap.get(V.FIT) == V.VM_PUT:
        now = console(SERVED)
        sent = json.loads((mine / "served.json").read_text()) if (mine / "served.json").exists() else {}
        served = json.loads(now.stdout) if not now.returncode else None
        bodies = [vm / "put" / f"{n}.json" for n in (V.FIT, "algorithm-policy")]
        if served is None:
            put = {k: f"not PUT: GET failed ({(now.stderr.strip().splitlines() or ['?'])[-1][:160]})" for k in ap}
        elif not art:
            put = {k: "not PUT: the run's record isn't in the store, so its tables aren't" for k in ap}
        elif any(served.get(k) != sent.get(k) for k in ("algorithm-policy", V.FIT)):
            put = {k: "not PUT: the route's served panel changed while the refresh ran" for k in ap}
        elif not all(b.exists() for b in bodies):
            put = {k: "not PUT: the run left no console body" for k in ap}
        else:
            r = console(PUT, *map(str, bodies))
            lines = [x for x in r.stdout.splitlines() if x.startswith("PUT ")]
            put = {u.rstrip("/").rsplit("/", 1)[-1]: s for u, s in (x[4:].split(": ", 1) for x in lines)}
            (mine / "vm-put.log").write_text("\n".join(lines) + "\n" + r.stderr)
        put = {k: v for k, v in put.items() if k in ("algorithm-policy", V.FIT)}
    posts = json.loads((vm / "post.json").read_text()) if (vm / "post.json").exists() else []
    for tag, text in posts:
        subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", tag, fix(text, put)], check=True)
    merged = None
    if (vm / "merge.bundle").exists():
        f = git("fetch", "-q", str(vm / "merge.bundle"), "refs/te-scorer/merge", cwd=RESCORE)
        p = git("push", "-q", "origin", f"FETCH_HEAD:refs/heads/{V.RESCORE_AHEAD}", cwd=RESCORE) if not f.returncode else f
        merged = "pushed" if not p.returncode else f"not pushed: {p.stderr.strip()[-200:]}"
    if (vm / "collect.json").exists():
        (LOOP / "missing.json").write_text(json.dumps(json.loads((vm / "collect.json").read_text())
                                                      .get("commits_missing") or []) + "\n")
    panels = ARTIFACTS / "te-panels"
    for f in (vm / "panels").glob("*") if (vm / "panels").exists() else []:
        panels.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, panels / f.name)
    if (vm / "te-night2-efficiency.md").exists():
        shutil.copyfile(vm / "te-night2-efficiency.md", ARTIFACTS / "te-night2-efficiency.md")
    stamp = (vm / "stamp").read_text().strip() if (vm / "stamp").exists() else None
    label = None
    if art and cycle:
        r = subprocess.run(["research", "data", "label", art, "note",
                            f"the scorer's refresh {stamp} on {NODE} ({run}): out/tables, te_core.rescore's tables and "
                            f"algorithm-policy's; out/fb, frontier-branches (benchmarks/covert_witness/scorer/"
                            f"refresh-node.sh)", "--by", "@training"],
                           capture_output=True, text=True, check=False, env=os.environ | {"AGENT_GUARD_LOCAL_OK": "1"})
        label = "labelled" if not r.returncode else f"not labelled: {r.stderr.strip()[-200:]}"
    done = {"run": run, "state": st, "record": art, "stamp": stamp, "put": put, "posted": [t for t, _ in posts],
            "merge": merged, "label": label, "art": cycle.get("art")}
    (LOOP / "last.json").write_text(json.dumps(done | {"vm_put": (mine / "vm-put.log").read_text()
                                                       if (mine / "vm-put.log").exists() else ""}, indent=1) + "\n")
    say("finished", json.dumps(done))
    return done


def once(slot: float) -> dict:
    last = json.loads((LOOP / "last.json").read_text()) if (LOOP / "last.json").exists() else {}
    if last.get("run") and state(last["run"]) not in V.ENDED:
        say("skipped: the last refresh", last["run"], "is still going")
        return {}
    say(push())
    r = console(SERVED)
    if r.returncode:
        say("served panels not read, so the run PUTs nothing:", (r.stderr.strip().splitlines() or ["?"])[-1][:200])
    LOOP.mkdir(parents=True, exist_ok=True)
    send, served, log = [], LOOP / "served.json", LOOP / "vm-put.log"
    served.unlink(missing_ok=True)
    if not r.returncode:
        served.write_text(r.stdout)
        send.append(served)
    # a refresh off the slots' grid (`once` by hand) stops its shards before the next slot, which then runs
    after = FIRST + ((slot - FIRST) // SLOT_S + 1) * SLOT_S
    env = {"TE_HERE_UNTIL": f"{min(slot + UNTIL_S, after - LATE_S, END):.0f}"}
    if last.get("vm_put") and last.get("stamp"):
        log.write_text(last["vm_put"])
        send.append(log)
        env["TE_VM_PUT_FOR"] = last["stamp"]
    run = launch(env, send, MEM_GB, TIMEOUT)
    (LOOP / run).mkdir(parents=True, exist_ok=True)
    if served.exists():
        shutil.copyfile(served, LOOP / run / "served.json")
    (LOOP / "last.json").write_text(json.dumps(last | {"run": run}, indent=1) + "\n")
    return finish(run, time.time() + POOL_WAIT_S + remote.timeout_seconds(TIMEOUT) + 600)


def loop() -> int:
    while True:
        now = time.time()
        slot = FIRST + max(0, -(-(now - LATE_S - FIRST) // SLOT_S)) * SLOT_S
        if slot >= END:
            say("done: no slot before", datetime.fromtimestamp(END, UTC).isoformat())
            return 0
        if slot > now:
            say("next slot", datetime.fromtimestamp(slot, UTC).strftime("%H:%MZ"))
            time.sleep(slot - now)
        try:
            once(slot)
        except (SystemExit, Exception) as e:  # noqa: BLE001 - a failed refresh is reported and the next slot runs
            say("refresh failed:", str(e)[:600])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("once")
    f = sub.add_parser("finish")
    f.add_argument("run")
    sub.add_parser("loop")
    a = ap.parse_args()
    if a.cmd == "init":
        return init()
    if a.cmd == "once":
        once(time.time())
        return 0
    if a.cmd == "finish":
        finish(a.run, time.time() + 1800)
        return 0
    return loop()


if __name__ == "__main__":
    sys.exit(main())
