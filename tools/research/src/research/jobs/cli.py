"""`research jobs` and `research worker`: the Job queue from the command line (Control's `/api/jobs`; `research.jobs.model` is its
reference).  The Job queue holds batch, unattended work on pods; an interactive session on a pod is a Broker allocation.

    research jobs add merge-check --commit REF [--base SHA] [--verdict-pack ART] [common]
    research jobs add lean-regen --commit REF --package DIR [--package DIR ...] [common]
    research jobs add audit-main [--commit REF] [common]            a cold check of main's tip (the nightly cron's job)
        common: [--priority N] [--timeout-s S] [--max-attempts N] [--dedupe-key K] [--line PREFIX] [--repo OWNER/NAME]
    research jobs list [--state S] [--kind K] [--limit N] [--json]
    research jobs show ID [--json]
    research jobs cancel ID
    research jobs events [--after ID] [--kind K] [--subject S] [--limit N] [--json]
    research worker --dispatch --pods A,B,... [--name rc] [--kinds merge-check,lean-regen] [--once] [--workdir DIR] [--token-file F]
                    [--repo OWNER/NAME]
    research worker --dispatch --kinds research-run [--slots 4] --name control     a lead's `research run` line (dispatch.py)

The tokens: `research jobs` uses `$JOBS_TOKEN`, else ~/.research/jobs/token (the coordinator's); `research worker` uses
--token-file, else `$JOBS_WORKER_TOKEN`, else ~/.research/jobs/dispatch-{name}.token (the dispatcher's), never the coordinator's.
`add` resolves --commit in the current checkout, and --base (for `merge_requires`) defaults to `main`'s tip on GitHub, read
with `git ls-remote` rather than a local ref that may be stale.  --timeout-s defaults to the dispatcher's DEFAULT_TIMEOUT_S, and
the job asks for a pod whose lease has that long left plus the lease margin (`min_lease_s`), so no check starts on a pod that
ends before it can finish.  A check asks for AVX-512 only when the tree says so, as preflight.py decides it: the change
touches what `merge_requires` sends the Lean agreement for, and `upstream.json`'s rustflags match preflight.py's AVX512 pattern
(read from the commit, never run).  Today's pin is `x86-64-v3`, which needs none.

The dedupe key makes adding idempotent, and a deliberate re-run names a new one (--dedupe-key).  A merge check's is
`merge-check:{commit}`.  A Lean regeneration's is `lean-regen:{digest of its inputs}`: the blobs, at that commit, of every tracked
file the audit reads (each package, the packages its `proved_in` names, and every package these require by path, less the
Lake packages nested in them, `tools/verity/lean/`, and the controls' packages every Lean check run builds, wherever
`check.FIXTURES` puts them; `tools/verity/lean/lean_audit.py`'s inputs without check's own files).  So candidates
whose Lean inputs are the same share one regeneration, and the second add answers with the first job and its run.  The site
is `$JOBS_URL`.  Exit 1 when the Job queue refuses, 4 when it can't be reached.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

from .client import WORKER_TOKEN_ENV, HTTPClient, Unreachable, load_token, worker_token_file
from .dispatch import AUDIT_TIMEOUT_S, DEFAULT_TIMEOUT_S, EMPTY_TREE, SHA, WORKDIR, Dispatcher, DispatchError, ResearchRunner, lease_needed
from .dispatch import KINDS as DISPATCH_KINDS
from .dispatch import POD_KINDS, RUN_KIND, RUN_SLOTS

REPO = "compute-verification/verity"
LEAN_TOOL = "tools/verity/lean/"
#: the Lean check, whose `FIXTURES` (the controls' packages, which every run of it builds) the key reads wherever they are
LEAN_CHECK = f"{LEAN_TOOL}check.py"
LEAN_KEY_SCHEMA = "lean-regen-inputs/v1"


def resolve_commit(ref: str) -> str:
    if SHA.fullmatch(ref):
        return ref
    r = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"research jobs: {ref!r} is not a commit here: {r.stderr.strip()}")
    return r.stdout.strip()


def main_tip(repo: str) -> str:
    r = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}.git", "refs/heads/main"], capture_output=True, text=True)
    if r.returncode or not r.stdout.split():
        raise SystemExit(f"research jobs: can't read main's tip on {repo} ({r.stderr.strip() or 'no answer'}); pass --base")
    return r.stdout.split()[0]


def _git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="surrogateescape")
    if r.returncode:
        raise SystemExit(f"research jobs: git {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def lean_inputs(commit: str, packages: list[str], cwd: Path | None = None) -> dict[str, str]:
    """{path: blob sha} of every tracked file `check.py --build --update PACKAGE...` reads at `commit` (module docstring)."""
    files = {}
    for entry in _git("ls-tree", "-r", "-z", "--full-tree", commit, cwd=cwd).split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            files[path] = meta.split()[2]
    def read(path: str) -> dict:
        return json.loads(_git("show", f"{commit}:{path}", cwd=cwd)) if path in files else {}

    def deps(pkg: str) -> set[str]:
        return {posixpath.normpath(posixpath.join(pkg, d["dir"]))
                for d in read(f"{pkg}/lake-manifest.json").get("packages", []) if d.get("type") == "path"}

    todo: list[str] = []
    for pkg in packages:
        pkg = pkg.strip("/")
        if f"{pkg}/lean-audit.json" not in files:
            raise SystemExit(f"research jobs: {pkg} has no lean-audit.json at {commit[:12]}")
        todo += [pkg, *(posixpath.normpath(posixpath.join(pkg, d)) for d in read(f"{pkg}/lean-audit.json").get("proved_in", []))]
    pkgs: set[str] = set()
    while todo:
        if (pkg := todo.pop()) not in pkgs:
            pkgs.add(pkg)
            todo += deps(pkg)

    def own(pkg: str) -> list[str]:
        nested = {f.rsplit("/", 1)[0] + "/" for f in files if f.startswith(pkg + "/") and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")
                  and f.rsplit("/", 1)[0] != pkg}
        return [f for f in files if f.startswith(pkg + "/") and not any(f.startswith(n) for n in nested)]

    tool = (LEAN_TOOL, lean_fixtures(commit, cwd) + "/")
    return {f: files[f] for f in sorted({*(f for p in pkgs for f in own(p)), *(f for f in files if f.startswith(tool))})}


def lean_fixtures(commit: str, cwd: Path | None = None) -> str:
    """`check.FIXTURES` at ``commit``, as a path in the tree, read without running check.py: its module-level assignments are
    evaluated in order where they are a path built from `Path(__file__)`, `.resolve()`, `.parent`, `.parents[i]`, `/` and
    string literals."""
    def ev(n: ast.expr, names: dict[str, PurePosixPath]) -> PurePosixPath | str | None:
        match n:
            case ast.Constant(value=str(s)):
                return s
            case ast.Name(id=name) if name in names:
                return names[name]
            case ast.Call(func=ast.Name(id="Path"), args=[ast.Name(id="__file__")]):
                return PurePosixPath(LEAN_CHECK)
            case ast.Call(func=ast.Attribute(value=v, attr="resolve"), args=[]):
                return ev(v, names)
            case ast.Attribute(value=v, attr="parent"):
                return p.parent if isinstance(p := ev(v, names), PurePosixPath) else None
            case ast.Subscript(value=ast.Attribute(value=v, attr="parents"), slice=ast.Constant(value=int(i))):
                return p.parents[i] if isinstance(p := ev(v, names), PurePosixPath) and 0 <= i < len(p.parents) else None
            case ast.BinOp(left=a, op=ast.Div(), right=b):
                l, r = ev(a, names), ev(b, names)
                return l / r if isinstance(l, PurePosixPath) and r is not None else None
        return None

    names: dict[str, PurePosixPath] = {}
    for node in ast.parse(_show(commit, LEAN_CHECK, cwd) or "").body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(value := ev(node.value, names), PurePosixPath):
            names[node.targets[0].id] = value
    if "FIXTURES" not in names:
        raise SystemExit(f"research jobs: {LEAN_CHECK} at {commit[:12]} sets no `FIXTURES` this reads as a path "
                         f"(`lean_fixtures`): build it from `Path(__file__)` or `HERE` and string literals")
    return str(names["FIXTURES"])


def lean_key(commit: str, packages: list[str], cwd: Path | None = None) -> str:
    h = hashlib.sha256(json.dumps([LEAN_KEY_SCHEMA, sorted(p.strip("/") for p in packages)]).encode())
    for path, blob in lean_inputs(commit, packages, cwd).items():
        h.update(f"{path}\0{blob}\0".encode())
    return h.hexdigest()


#: preflight.py's AVX512, for a tree from before it
AVX512_BEFORE_PREFLIGHT = r"target-cpu=(native|x86-64-v4)\b|avx512"
UPSTREAM = "verity/protocols/proofs/flock/prover/agreement/upstream.json"


def _show(commit: str, path: str, cwd: Path | None = None) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=cwd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _literal(source: str | None, name: str):
    """The literal a module assigns to ``name`` (or passes as keyword ``name`` in a call), read without running the module."""
    for node in ast.walk(ast.parse(source or "")):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and value.args:                 # AVX512 = re.compile(r"...")
                value = value.args[0]
            return ast.literal_eval(value)
        if isinstance(node, ast.keyword) and node.arg == name:
            return ast.literal_eval(node.value)
    return None


def needs_avx512(commit: str, base: str, cwd: Path | None = None) -> bool:
    """Whether a check of `commit` against `base` runs the Lean agreement with an upstream build that needs AVX-512."""
    pin = _show(commit, UPSTREAM, cwd)
    if pin is None:
        return False
    flags = json.loads(pin)["upstream"].get("rustflags") or "target-cpu=native"
    pattern = _literal(_show(commit, "tools/check/preflight.py", cwd), "AVX512") or AVX512_BEFORE_PREFLIGHT
    if not re.search(pattern, flags):
        return False
    rule = (_literal(_show(commit, "tools/check/tool.py", cwd), "merge_requires") or {}).get("lean-agreement")
    from ..merge import rule_touches

    changed = [p for p in _git("diff", "--no-renames", "--name-only", "-z", base, commit, cwd=cwd).split("\0") if p]   # a rename's source too
    return bool(rule and rule_touches(rule, changed))


def add_body(ns: argparse.Namespace) -> dict:
    if ns.commit is None and ns.kind != "audit-main":
        raise SystemExit(f"research jobs add {ns.kind}: --commit is required")
    commit = resolve_commit(ns.commit) if ns.commit else main_tip(ns.repo)
    timeout = ns.timeout_s or (AUDIT_TIMEOUT_S if ns.kind == "audit-main" else DEFAULT_TIMEOUT_S)
    payload: dict = {"repo": ns.repo, "commit": commit, "timeout_s": timeout}
    requirements: dict = {"min_lease_s": lease_needed(timeout)}
    if ns.kind == "merge-check":
        payload["base"] = resolve_commit(ns.base) if ns.base else main_tip(ns.repo)
        payload["verdict_pack"] = ns.verdict_pack
        if needs_avx512(commit, payload["base"]):
            requirements["avx512"] = True
        key = f"merge-check:{commit}"
    elif ns.kind == "audit-main":
        if needs_avx512(commit, EMPTY_TREE):
            requirements["avx512"] = True
        key = f"audit-main:{commit}"
    else:
        if not ns.package:
            raise SystemExit("research jobs add lean-regen: name at least one --package")
        payload["packages"] = ns.package
        payload["inputs"] = lean_key(commit, ns.package)
        key = f"lean-regen:{payload['inputs']}"
    priority = ns.priority if ns.priority is not None else (5 if ns.kind == "audit-main" else 0)
    body = {"kind": ns.kind, "payload": payload, "requirements": requirements, "priority": priority,
            "dedupe_key": ns.dedupe_key or key}
    if ns.max_attempts:
        body["max_attempts"] = ns.max_attempts
    if ns.line:
        body["line"] = ns.line
    return body


def _row(j: dict) -> str:
    p = j.get("payload") or {}
    where = j.get("lease_slot") or j.get("lease_worker") or ""
    return (f"{j['id']:>6}  {j['kind']:<12} {j['state']:<9} {j['priority']:>4}  {j['attempts']}/{j['max_attempts']}  "
            f"{str(p.get('commit', ''))[:12]:<12}  {where:<14} {j['enqueued_at']}  {j['enqueued_by']}")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="research jobs", description="the Job queue (research/jobs/cli.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("kind", choices=("merge-check", "lean-regen", "audit-main"))
    add.add_argument("--commit")
    add.add_argument("--base")
    add.add_argument("--verdict-pack")
    add.add_argument("--package", action="append", default=[])
    add.add_argument("--priority", type=int, default=None, help="default 0, and 5 for audit-main")
    add.add_argument("--timeout-s", type=int)
    add.add_argument("--max-attempts", type=int)
    add.add_argument("--dedupe-key")
    add.add_argument("--line", help="the budget line the job's compute is charged to, e.g. vy-coord-")
    add.add_argument("--repo", default=REPO)
    ls = sub.add_parser("list")
    ls.add_argument("--state")
    ls.add_argument("--kind")
    ls.add_argument("--limit", type=int, default=50)
    ls.add_argument("--json", action="store_true")
    show = sub.add_parser("show")
    show.add_argument("id", type=int)
    show.add_argument("--json", action="store_true")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("id", type=int)
    ev = sub.add_parser("events")
    ev.add_argument("--after", type=int, default=0)
    ev.add_argument("--kind")
    ev.add_argument("--subject")
    ev.add_argument("--limit", type=int, default=100)
    ev.add_argument("--json", action="store_true")
    return ap


def _query(**kw) -> str:
    q = "&".join(f"{k}={v}" for k, v in kw.items() if v not in (None, ""))
    return f"?{q}" if q else ""


def main(argv: list[str], client=None) -> int:
    ns = _parser().parse_args(argv)
    c = client or HTTPClient()
    try:
        if ns.cmd == "add":
            status, body = c.call("POST", "/api/jobs", add_body(ns))
            if status in (200, 201):
                print(f"job {body['id']} {'added' if body['created'] else 'already there'} ({body['state']})")
                if not body["created"]:
                    st, j = c.call("GET", f"/api/jobs/{body['id']}")
                    if st == 200:
                        run = (j.get("result") or {}).get("run_id")
                        print(f"    added for commit {j['payload'].get('commit')}" + (f"; run {run}" if run else ""))
                return 0
        elif ns.cmd == "list":
            status, body = c.call("GET", "/api/jobs" + _query(state=ns.state, kind=ns.kind, limit=ns.limit))
            if status == 200:
                print(json.dumps(body["jobs"], indent=1) if ns.json else "\n".join(_row(j) for j in body["jobs"]) or "no jobs")
                return 0
        elif ns.cmd == "show":
            status, body = c.call("GET", f"/api/jobs/{ns.id}")
            if status == 200:
                if ns.json:
                    print(json.dumps(body, indent=1))
                else:
                    print(_row(body))
                    print(f"        payload {json.dumps(body['payload'])}  requirements {json.dumps(body['requirements'])}")
                    if body.get("result"):
                        print(f"        result  {json.dumps(body['result'])}")
                    for a in body.get("history", []):
                        print(f"        attempt {a['n']}: {a['worker']} {a.get('slot') or ''} {a['claimed_at']} -> "
                              f"{a.get('outcome') or 'running'} {a.get('run_id') or ''} {a.get('detail') or ''}")
                return 0
        elif ns.cmd == "cancel":
            status, body = c.call("POST", f"/api/jobs/{ns.id}/cancel", {})
            if status == 200:
                print(f"job {ns.id}: {body['state']}" + (" (the worker stops it at its next renewal)" if body.get("cancel_requested") else ""))
                return 0
        else:
            status, body = c.call("GET", "/api/events" + _query(after=ns.after, kind=ns.kind, subject=ns.subject, limit=ns.limit))
            if status == 200:
                if ns.json:
                    print(json.dumps(body, indent=1))
                else:
                    for e in body["events"]:
                        print(f"{e['id']:>7}  {e['at']}  {e['kind']:<16} {e['subject']:<10} {e['actor']:<14} {json.dumps(e['payload'])}")
                    print(f"next {body['next']}")
                return 0
    except Unreachable as e:
        print(f"research jobs: the Job queue is unreachable: {e}", file=sys.stderr)
        return 4
    print(f"research jobs: {status} {body.get('error', body)}", file=sys.stderr)
    return 1


def worker_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research worker", description="step 1's dispatcher (research/jobs/dispatch.py)")
    ap.add_argument("--dispatch", action="store_true", help="claim jobs and run them with `research run --on` onto --pods")
    ap.add_argument("--pods", default="", help="machines.toml names, comma-separated; one job per pod at a time")
    ap.add_argument("--slots", type=int, default=RUN_SLOTS, help=f"{RUN_KIND} jobs at a time, when it is among --kinds")
    ap.add_argument("--name", default="rc", help="the worker is dispatch:{name}")
    ap.add_argument("--kinds", default=",".join(POD_KINDS), help=f"of {', '.join(DISPATCH_KINDS)}")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--workdir", default=str(WORKDIR))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--token-file", type=Path, help="the dispatcher's token (default: $JOBS_WORKER_TOKEN, else WORKDIR/dispatch-NAME.token)")
    ns = ap.parse_args(argv)
    if not ns.dispatch:
        print("research worker: only --dispatch exists until pull workers (step 2)", file=sys.stderr)
        return 2
    kinds = tuple(k for k in ns.kinds.split(",") if k)
    if set(kinds) - set(DISPATCH_KINDS):
        print(f"research worker: the dispatcher runs only {', '.join(DISPATCH_KINDS)}", file=sys.stderr)
        return 2
    pods = [p for p in ns.pods.split(",") if p]
    runner = ResearchRunner(repo=ns.repo, workdir=Path(ns.workdir).expanduser())
    why = ("--pods names no pod for " + ", ".join(sorted(set(kinds) - {RUN_KIND})) if set(kinds) - {RUN_KIND} and not pods else
           f"--slots must be at least 1 for {RUN_KIND}" if RUN_KIND in kinds and ns.slots < 1 else
           f"{RUN_KIND} runs `research run` from the dispatcher's own checkout, and this one runs from none"
           if RUN_KIND in kinds and runner.launcher_tree is None else None)
    if why:
        print(f"research worker: {why}", file=sys.stderr)
        return 2
    from .. import remote

    try:
        for p in pods:
            remote.resolve_machine(p, "verity")
    except remote.Refused as e:
        print(f"research worker: {e}", file=sys.stderr)
        return 2
    workdir = runner.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    lock = open(workdir / f"dispatch-{ns.name}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"research worker: dispatch:{ns.name} already runs from {workdir}", file=sys.stderr)
        return 2
    token = (load_token({}, ns.token_file) if ns.token_file else
             load_token(path=worker_token_file(ns.name, workdir), var=WORKER_TOKEN_ENV))
    client = HTTPClient(token=token)
    d = Dispatcher(client, runner, pods, name=ns.name, kinds=kinds, run_slots=ns.slots, repo=ns.repo,
                   state_path=workdir / f"dispatch-{ns.name}.json", out=lambda m: print(m, flush=True))
    d.log(f"{d.worker} on {client.url}: {', '.join(pods) or 'no pods'}; {', '.join(kinds)}"
          + (f"; {ns.slots} {RUN_KIND} slots, launched from {runner.launcher_tree}" if RUN_KIND in kinds else ""))
    try:
        d.run(once=ns.once)
    except DispatchError as e:
        d.log(f"stopping: {e}")
        return 1
    except KeyboardInterrupt:
        d.log("stopping; the runs in flight keep going, and a restart picks them up from the state file")
        return 130
    return 0
