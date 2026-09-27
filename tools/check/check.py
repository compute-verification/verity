"""`check`: the repository's one check, what a merge into `main` requires (`research merge`).

    uv run python tools/check/check.py                      run every step here, in the checkout
    uv run python tools/check/check.py --record [--on M]    the same through `research run --tool check` (on machine M, a
                                                            multi-core CPU pod): an Attempt of the clean commit checked out, the
                                                            record `research merge` reads; with the pinned upstream build and
                                                            inputs when the commit changes backends/flock/ against --base REF
                                                            (default origin/main)

Three independent groups run at once, the cores split between them (`--cores N`, default the CPUs this process may use):

    pytest          every test under the root `pyproject.toml` testpaths except the `circuit_suite` ones, `-n` half the cores
    circuit-check   `circuit-check --all --jobs` the other half: every registered Definition and template (`tools/circuit_check`),
                    its report beside the run's result
    lean            `lake build` of the Lean verifier (PR #85), then its partition check against `verity.ir.partition`
                    (`unit_cut_agree.py`), then its agreement with upstream's Rust verifier (`ci.py`) on the upstream build
                    stored once and pinned by hash (`backends/flock/verifier/upstream.json`), when the run has it; skipped by
                    name without it.  `research merge` requires it for a change under backends/flock/ (`tool.py`)

Caches (``$CHECK_CACHE``, default ``~/.cache/verity-check``; ``--no-cache`` for a cold run), each a hit only on identical inputs:

    pytest          a passing result is reused when the tree's every file (`tree_digest`), the steps' environment, the interpreter,
                    the installed distributions and the pytest arguments are the same (the tests start processes of their own,
                    so what they read cannot be traced: the whole tree is the input)
    circuit-check   `circuit-check --cache`: a report is reused when every file, directory listing and existence probe the
                    recorded run made is unchanged, with the checker's own version (`circuit_check.cache`)
    lean-agreement  a pass is reused when everything the job reads is the same (`agreement_key`: backends/flock/verifier/'s
                    tracked files, which pin the upstream build and inputs, the interpreter, the environment)

Inside a recorded run it writes `$RESEARCH_RUN_DIR/result.json` (kind `check/v1`): each step's status, seconds and cache use,
with each step's log beside it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: PR #85: the Lean verifier of record, and `ci.py`, its agreement job against upstream's Rust verifier
FLOCK_VERIFIER = ROOT / "backends" / "flock" / "verifier"
#: environment variables that say nothing about what a step computes (the shell's own bookkeeping)
VOLATILE = frozenset({"_", "OLDPWD", "PWD", "SHLVL", "TERM", "COLUMNS", "LINES", "TMUX", "TMUX_PANE", "SSH_CLIENT", "SSH_CONNECTION",
                      "SSH_TTY", "SSH_AUTH_SOCK", "XDG_SESSION_ID", "MAIL", "LOGNAME", "USER"})


def cache_dir() -> Path:
    return Path(os.environ.get("CHECK_CACHE") or Path.home() / ".cache" / "verity-check")


def cores() -> int:
    """The CPUs this process may use, bounded by the cgroup quota (a pod's `nproc` can exceed it)."""
    n = len(os.sched_getaffinity(0))
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            n = min(n, max(1, int(q) // int(p)))
    except (OSError, ValueError):
        pass
    return n


#: the toolchain `lean-toolchain` pins, through elan when `lake` is missing (as `ci-pod.sh` does)
ELAN = ('command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || { curl -sSf '
        'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }; '
        'export PATH="$HOME/.elan/bin:$PATH"; ')


#: upstream's verifier builds and the sets' stored inputs, stored once and pinned by hash (`upstream-build.sh`)
UPSTREAM_PIN = FLOCK_VERIFIER / "upstream.json"
#: where check looks for them besides the run's `inputs/` (`--send`): a local recorded run's launcher sets it
AGREEMENT_DIR_ENV = "FLOCK_AGREEMENT_INPUTS"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def agreement_files(out: Path) -> dict[str, Path]:
    """The pinned upstream build and inputs among the files sent with the run (`$RESEARCH_RUN_DIR/inputs/`) or in
    ``$FLOCK_AGREEMENT_INPUTS``, found by sha256: {"upstream": path, "inputs": path} for those present."""
    pin = json.loads(UPSTREAM_PIN.read_text())
    want = {pin[k]["sha256"]: k for k in ("upstream", "inputs")}
    dirs = [out / "inputs", *([Path(os.environ[AGREEMENT_DIR_ENV])] if os.environ.get(AGREEMENT_DIR_ENV) else [])]
    found = {}
    for p in (q for d in dirs if d.is_dir() for q in sorted(d.iterdir()) if q.is_file()):
        if p.stat().st_size and (k := want.get(_sha256(p))):
            found[k] = p
    return found


def lean_steps(out: Path) -> list[tuple[str, list[str] | str, Path]]:
    """The Lean steps: nothing until PR #85 is on the tree (a skip, named); then `lake build`, the Lean partition check's
    conformance with `verity.ir.partition` (`unit_cut_agree.py`), and the agreement job against upstream's Rust verifier on the
    pinned upstream build (`upstream.json`) when the run has it; without it that step is skipped by name.  `research merge`
    requires it for a change under `backends/flock/` (`tools/check/tool.py`), and `--record` sends it then."""
    lean = FLOCK_VERIFIER / "lean"
    if not (lean / "lakefile.toml").exists():
        return [("lean", "skipped: PR #85 (backends/flock/verifier) is not on this tree", ROOT)]
    verify = lean / ".lake" / "build" / "bin" / "flock-verify"
    agreement: list[str] | str = ("skipped: the pinned upstream build was not sent (a merge changing backends/flock/ requires "
                                  "this step: `check.py --record` sends it)")
    if UPSTREAM_PIN.exists():
        got = agreement_files(out)
        if len(got) == 2:
            agreement = [sys.executable, str(Path(__file__).resolve()), "--lean-agreement", str(out), str(got["upstream"]), str(got["inputs"])]
        elif got:
            agreement = f"only {', '.join(got)} of the pinned upstream build and inputs was sent"
    return [("lean-build", ["bash", "-c", ELAN + "lake build"], lean),
            ("lean-unit-cut", [sys.executable, str(FLOCK_VERIFIER / "unit_cut_agree.py"), "--lean", str(verify),
                               "--out", str(out / "lean-unit-cut")], ROOT),
            ("lean-agreement", agreement, ROOT)]


def memory_gb() -> int:
    """The memory this process may use: the cgroup limit when there is one (a pod's /proc/meminfo is the host's), else
    MemAvailable."""
    try:
        m = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if m != "max":
            return int(m) >> 30
    except (OSError, ValueError):
        pass
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) >> 20
    return 0


def session_jobs() -> int:
    """`ci.py --session-jobs`: its own default counts the host's CPUs and memory inside a container, so check passes what this
    process may use, at ci.py's `GB_PER_PROCESS` per verifier process."""
    m = re.search(r"^GB_PER_PROCESS = (\d+)", (FLOCK_VERIFIER / "ci.py").read_text(), re.M)
    return max(1, min(cores(), memory_gb() // int(m.group(1) if m else 8)))


def lean_agreement(out: Path, upstream: Path, inputs: Path) -> int:
    """Unpack the pinned upstream build and inputs, check every binary against the pin, and run `ci.py` on every replayable set."""
    import tarfile

    pin = json.loads(UPSTREAM_PIN.read_text())
    w = out / "flock-agreement"
    for name, src in (("upstream", upstream), ("inputs", inputs)):
        if _sha256(src) != pin[name]["sha256"]:
            print(f"check: {src} is not the pinned {name} ({pin[name]['art']})", file=sys.stderr)
            return 1
        with tarfile.open(src) as t:
            t.extractall(w / name, filter="data")
    ups = []
    for c, h in sorted(pin["upstream"]["binaries"].items()):
        b = w / "upstream" / f"upstream-{c}"
        if not b.is_file() or _sha256(b) != h:
            print(f"check: upstream-{c} does not match the pin", file=sys.stderr)
            return 1
        b.chmod(0o755)
        ups += ["--upstream", f"{c}={b}"]
    ci = pin.get("ci", {})
    sets = json.loads((FLOCK_VERIFIER / "vectors.json").read_text())["replayable"]["sets"]    # ci.py --jobs: sets at once
    argv = [sys.executable, str(FLOCK_VERIFIER / "ci.py"), "--lean", str(FLOCK_VERIFIER / "lean" / ".lake" / "build" / "bin" / "flock-verify"),
            *ups, "--work", str(w / "work"), "--inputs", str(w / "inputs"), "--jobs", str(max(1, min(cores(), len(sets)))),
            "--session-jobs", str(session_jobs()),
            "--seed", str(ci.get("seed", 20260926)), "--fuzz", str(ci.get("fuzz", 20)), "--out", str(w)]
    return subprocess.run(argv, cwd=ROOT).returncode


def record(argv: list[str]) -> int:
    """`--record [--on MACHINE] [--base REF]`: `research run --tool check` on this clean commit, with the pinned upstream build and
    inputs (fetched from the store) when the commit changes a path `tools/check/tool.py` names against REF (default origin/main)."""
    on = base = None
    rest = []
    it = iter(argv)
    for x in it:
        if x == "--on":
            on = next(it)
        elif x == "--base":
            base = next(it)
        else:
            rest.append(x)
    spec = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    base = base or "origin/main"
    changed = subprocess.run(["git", "diff", "--name-only", base, "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    need = [s for s, ps in tool.CHECK.merge_requires.items() if any(c.startswith(p) for p in ps for c in changed)]
    cmd = [sys.executable, "-m", "research", "run", "--tool", "check", "--project", "verity", "--cwd", "source"]
    env = dict(os.environ)
    if "lean-agreement" in need:
        from research.store.local import LocalStore

        pin = json.loads(UPSTREAM_PIN.read_text())
        store = LocalStore()
        files = {k: Path(store.fetch(pin[k]["art"])) for k in ("upstream", "inputs")}
        print(f"check: changes under {', '.join(tool.CHECK.merge_requires['lean-agreement'])} against {base}: sending the pinned "
              f"upstream build and inputs")
        cmd += [x for k in ("upstream", "inputs") for x in ("--input", f"flock_{k}={pin[k]['art']}")]
        if on:
            cmd += [x for f in files.values() for x in ("--send", str(f))]
        else:
            d = Path(tempfile.mkdtemp(prefix="check-agreement-"))
            for k, f in files.items():
                (d / f"{k}.tar.gz").symlink_to(f)
            env[AGREEMENT_DIR_ENV] = str(d)
    if on:
        cmd += ["--on", on, "--source", "."]
    cmd += ["--", *tool.CHECK.command, *rest]
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def groups(out: Path, n: int, cache: bool = True) -> list[list[tuple[str, list[str] | str, Path]]]:
    """The independent groups, each a sequence of (name, argv or the reason it is skipped / cannot run, cwd)."""
    py = sys.executable
    a = max(1, n // 2)
    b = max(1, n - a)
    pytest = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not circuit_suite", *(["-n", str(a)] if a > 1 else [])]
    cc = [py, "-m", "circuit_check", "--all", "-q", "--jobs", str(b), "--out", str(out / "circuit-check.json"),
          *(["--cache", str(cache_dir() / "circuit-check")] if cache else [])]
    return [[("pytest", pytest, ROOT)], [("circuit-check", cc, ROOT)], lean_steps(out)]


def steps(out: Path, n: int = 1) -> list[tuple[str, list[str] | str, Path]]:
    return [s for g in groups(out, n) for s in g]


def env() -> dict[str, str]:
    """The steps' environment, as a plain shell's: none of what the recording run gives its workload (`RESEARCH_*`, the
    machine's PYTHONPYCACHEPREFIX; the tests start runs of their own, which would inherit them), PYTHONPATH entries inside this
    tree only (a machine's runner puts its shipped `research` first), and a git identity when none is set."""
    out = {k: v for k, v in os.environ.items() if not k.startswith("RESEARCH_") and k != "PYTHONPYCACHEPREFIX"}
    keep = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p).resolve().is_relative_to(ROOT)]
    if keep:
        out["PYTHONPATH"] = os.pathsep.join(keep)
    else:
        out.pop("PYTHONPATH", None)
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):          # tests commit in throwaway repos; a fresh machine has no identity
        out.setdefault(k, "check")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        out.setdefault(k, "check@localhost")
    return out


def tracked_files(root: Path = ROOT) -> list[str] | None:
    """The tree's files as git has them: tracked and untracked-unignored in a checkout; in a shipped tree (no `.git`), the
    commit's tree as the machine's bare repository holds it (`READY.json` names it), never what steps have built in it since.
    None when neither is available."""
    if (root / ".git").exists():
        argv = ["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard"]
    else:
        ready, bare = root / "READY.json", root.parent.parent / "git" / "verity.git"
        if not (ready.exists() and bare.exists()):
            return None
        argv = ["git", "--git-dir", str(bare), "ls-tree", "-r", "-z", "--name-only", json.loads(ready.read_text())["tree"]]
    r = subprocess.run(argv, capture_output=True)
    return sorted(p.decode() for p in r.stdout.split(b"\0") if p) if r.returncode == 0 else None


def tree_digest(root: Path = ROOT) -> str | None:
    """sha256 over the tree's files (`tracked_files`: path and current content, a deleted one marked): what the tests read.
    None when the tree's files cannot be listed (then nothing is cached)."""
    files = tracked_files(root)
    if files is None:
        return None
    h = hashlib.sha256()
    for f in files:
        p = root / f
        h.update(f.encode() + b"\0" + (_sha256(p).encode() if p.is_file() and not p.is_symlink() else
                                        (os.readlink(p).encode() if p.is_symlink() else b"-")) + b"\0")
    return h.hexdigest()


def _env_key(e: dict[str, str]) -> dict[str, str]:
    """The environment as a key: without the shell's bookkeeping, and with this tree's path written `<tree>` (a machine keeps each
    commit's tree at its own path, and `uv run` puts that path in PATH, VIRTUAL_ENV and PYTHONPATH)."""
    return {k: v.replace(str(ROOT), "<tree>") for k, v in sorted(e.items()) if k not in VOLATILE}


def pytest_key(argv: list[str], e: dict[str, str]) -> str | None:
    from importlib import metadata

    tree = tree_digest()
    if tree is None:
        return None
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    doc = {"tree": tree, "argv": [a for a in argv[1:] if not a.isdigit() and a != "-n"],
           "env": _env_key(e), "python": sys.version, "platform": platform.platform(),
           "distributions": dists}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def agreement_key(e: dict[str, str]) -> str | None:
    """What the agreement job reads: the tracked files under backends/flock/verifier/ (the Lean sources and toolchain pin, `ci.py`
    and the scripts it runs, which import only the standard library and each other, `upstream.json` pinning the upstream build,
    the inputs and the job's settings), the interpreter and the platform.  None when the tree's files cannot be listed."""
    files = tracked_files()
    if files is None:
        return None
    h = hashlib.sha256(json.dumps([sys.version, platform.platform(), _env_key(e)]).encode())
    for f in agreement_closure(files):
        p = ROOT / f
        h.update(f.encode() + b"\0" + (_sha256(p).encode() if p.is_file() else b"-") + b"\0")
    return h.hexdigest()


def agreement_closure(files: list[str]) -> list[str]:
    """The tracked files the agreement job can read: backends/flock/verifier/, less the Lake packages nested in its Lean package
    (`level3/`, `soundness/`: proofs with a lakefile of their own), which `flock-verify` is not built from as long as the Lean
    package requires nothing (a `require` in its lakefile keeps them all in), and less that package's `lean-audit.json`, which
    only `tools/lean/audit.py` reads (a test fails if the agreement's scripts or the package's lakefile mention it)."""
    pre, lean = "backends/flock/verifier/", "backends/flock/verifier/lean/"
    lakefiles = [f for f in files if f.startswith(lean) and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")]
    root = [f for f in lakefiles if f.count("/") == lean.count("/")]
    nested = tuple(f.rsplit("/", 1)[0] + "/" for f in lakefiles if f not in root)
    if any("require" in (ROOT / f).read_text() for f in root if (ROOT / f).is_file()):
        nested = ()
    return [f for f in files if f.startswith(pre) and f != lean + "lean-audit.json" and not f.startswith(nested or ("\0",))]


def _step(name: str, argv: list[str] | str, cwd: Path, out: Path, cache: bool) -> dict:
    if isinstance(argv, str):
        skipped = argv.startswith("skipped:")
        print(f"check: {name}: {argv}", flush=True)
        return {"step": name, "status": "skipped" if skipped else "failed", "detail": argv, "seconds": 0.0}
    e = env()
    t0 = time.monotonic()
    hit = None
    key = (pytest_key(argv, e) if name == "pytest" else agreement_key(e) if name == "lean-agreement" else None) if cache else None
    if key:
        entry = cache_dir() / name / f"{key}.json"
        if entry.exists():
            hit = key
    if hit:
        rec = {"step": name, "status": "passed", "rc": 0, "cache": {"hit": hit}}
    else:
        log = out / f"{name}.log"
        with open(log, "wb") as f:
            rc = subprocess.run(argv, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT).returncode
        rec = {"step": name, "status": "passed" if rc == 0 else "failed", "rc": rc, "log": log.name}
        if rc == 0 and key:
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(json.dumps({"key": key, "argv": argv, "passed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            rec["cache"] = {"stored": key}
        if name == "circuit-check" and (out / "circuit-check.json").exists():
            rec["cache"] = json.loads((out / "circuit-check.json").read_text()).get("cache")
        if rc:
            tail = log.read_bytes().decode(errors="replace").splitlines()[-40:]
            print("\n".join(f"  {name}| {line}" for line in tail), flush=True)
    rec["seconds"] = round(time.monotonic() - t0, 1)
    print(f"check: {name}: {rec['status']}{' (cached)' if hit else ''} in {rec['seconds']} s", flush=True)
    return rec


def run(out: Path, n: int | None = None, cache: bool = True) -> tuple[bool, list[dict]]:
    n = n or cores()

    def group(g: list) -> list[dict]:
        done = []
        for name, argv, cwd in g:
            done.append(_step(name, argv, cwd, out, cache))
            if done[-1]["status"] == "failed":
                break
        return done

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = [r for rs in ex.map(group, groups(out, n, cache)) for r in rs]
    return not any(r["status"] == "failed" for r in results), results


def write_result(run_dir: Path, ok: bool, done: list[dict]) -> None:
    doc = {"schema": "research/result/v0.1", "kind": "check/v1", "run_id": os.environ["RESEARCH_RUN_ID"],
           "workload_fingerprint": {"steps": [s["step"] for s in done]},
           "validation": {"status": "passed" if ok else "failed", "checker": "tools/check/check.py",
                          "detail": "; ".join(f"{s['step']} {s['status']}" for s in done)},
           "measurements": [{"name": f"{s['step']}_wall", "value": s["seconds"], "unit": "s"} for s in done],
           "steps": done, "artifacts": [p for p in ("circuit-check.json", *(s.get("log") for s in done)) if p and (run_dir / p).exists()]}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def main(argv: list[str]) -> int:
    if argv == ["--agreement-files"]:
        from research.store.local import LocalStore

        pin = json.loads(UPSTREAM_PIN.read_text())
        print(" ".join(f"--send {LocalStore().fetch(pin[k]['art'])}" for k in ("upstream", "inputs")))
        return 0
    if argv[:1] == ["--lean-agreement"] and len(argv) == 4:
        return lean_agreement(*map(Path, argv[1:]))
    if "--record" in argv:
        if os.environ.get("RESEARCH_RUN_ID"):
            print("check: --record inside a recorded run", file=sys.stderr)
            return 2
        return record([a for a in argv if a != "--record"])
    cache = "--no-cache" not in argv
    rest = [a for a in argv if a != "--no-cache"]
    n = None
    if rest[:1] == ["--cores"] and len(rest) == 2 and rest[1].isdigit():
        n, rest = int(rest[1]), []
    if rest:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    run_dir = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else None
    out = run_dir or Path(tempfile.mkdtemp(prefix="check-"))
    t0 = time.monotonic()
    ok, done = run(out, n, cache)
    if run_dir is not None:
        write_result(run_dir, ok, done)
    print(f"check: {'passed' if ok else 'FAILED'} in {time.monotonic() - t0:.1f} s ({', '.join(s['step'] + ' ' + s['status'] for s in done)});"
          f" outputs in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
