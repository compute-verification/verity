"""`check`: the repository's one check, what a merge into `main` requires (`research merge`).

    uv run python tools/check/check.py              run every step here, in the checkout
    uv run python tools/check/check.py --record     the same through `research run --tool check`: published as an Attempt of the
                                                    commit checked out (a clean tree), the record `research merge` reads
    research run --on MACHINE --project verity --source . --cwd source --tool check -- uv run --locked python tools/check/check.py
                                                    the same on a machine (a multi-core CPU pod)

Three independent groups run at once, the cores split between them (`--cores N`, default the CPUs this process may use):

    pytest          every test under the root `pyproject.toml` testpaths except the `circuit_suite` ones, `-n` half the cores
    circuit-check   `circuit-check --all --jobs` the other half: every registered Definition and template (`tools/circuit_check`),
                    its report beside the run's result
    lean            the Lean verifier's build, then its agreement with upstream's Rust verifier (`lean_steps`): skipped until PR #85
                    puts `backends/flock/verifier/lean` on the tree, required from then on

Caches (``$CHECK_CACHE``, default ``~/.cache/verity-check``; ``--no-cache`` for a cold run), each a hit only on identical inputs:

    pytest          a passing result is reused when the tree's every file (`tree_digest`), the steps' environment, the interpreter,
                    the installed distributions and the pytest arguments are the same (the tests start processes of their own,
                    so what they read cannot be traced: the whole tree is the input)
    circuit-check   `circuit-check --cache`: a report is reused when every file, directory listing and existence probe the
                    recorded run made is unchanged, with the checker's own version (`circuit_check.cache`)

Inside a recorded run it writes `$RESEARCH_RUN_DIR/result.json` (kind `check/v1`): each step's status, seconds and cache use,
with each step's log beside it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
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


def lean_steps(out: Path) -> list[tuple[str, list[str] | str, Path]]:
    """The Lean step: nothing until PR #85 is on the tree (a skip, named); then `lake build` and the agreement job over its
    replayable sets, with upstream's verifier builds from ``FLOCK_UPSTREAM`` (``PR83_COMMIT=BIN[,...]``, as `ci.py --upstream`).
    A missing toolchain or ``FLOCK_UPSTREAM`` fails the step: once the verifier is on the tree, `check` requires it."""
    lean = FLOCK_VERIFIER / "lean"
    if not (lean / "lakefile.toml").exists():
        return [("lean", "skipped: PR #85 (backends/flock/verifier) is not on this tree", ROOT)]
    upstream = [u for u in os.environ.get("FLOCK_UPSTREAM", "").split(",") if u]
    agree = [sys.executable, str(FLOCK_VERIFIER / "ci.py"), "--lean", str(lean / ".lake" / "build" / "bin" / "flock-verify"),
             *[x for u in upstream for x in ("--upstream", u)], "--work", str(out / "flock-agreement")]
    return [("lean-build", ["lake", "build"], lean),
            ("lean-agreement", agree if upstream else "FLOCK_UPSTREAM is not set: the agreement job needs upstream's verifier builds", ROOT)]


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


def tree_digest(root: Path = ROOT) -> str:
    """sha256 over every file under ``root`` (path and content), outside `.git`, `.venv` and bytecode: the tree the tests read."""
    h = hashlib.sha256()
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in (".git", ".venv", "__pycache__"))
        for f in sorted(files):
            p = Path(d) / f
            if p.is_file() and not p.is_symlink():
                h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def pytest_key(argv: list[str], e: dict[str, str]) -> str:
    from importlib import metadata

    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    doc = {"tree": tree_digest(), "argv": [a for a in argv[1:] if not a.isdigit() and a != "-n"],
           "env": {k: v for k, v in sorted(e.items()) if k not in VOLATILE}, "python": sys.version, "platform": platform.platform(),
           "distributions": dists}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _step(name: str, argv: list[str] | str, cwd: Path, out: Path, cache: bool) -> dict:
    if isinstance(argv, str):
        skipped = argv.startswith("skipped:")
        print(f"check: {name}: {argv}", flush=True)
        return {"step": name, "status": "skipped" if skipped else "failed", "detail": argv, "seconds": 0.0}
    e = env()
    t0 = time.monotonic()
    hit = None
    if cache and name == "pytest":
        key = pytest_key(argv, e)
        entry = cache_dir() / "pytest" / f"{key}.json"
        if entry.exists():
            hit = key
    if hit:
        rec = {"step": name, "status": "passed", "rc": 0, "cache": {"hit": hit}}
    else:
        log = out / f"{name}.log"
        with open(log, "wb") as f:
            rc = subprocess.run(argv, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT).returncode
        rec = {"step": name, "status": "passed" if rc == 0 else "failed", "rc": rc, "log": log.name}
        if rc == 0 and cache and name == "pytest":
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
    if "--record" in argv:
        if os.environ.get("RESEARCH_RUN_ID"):
            print("check: --record inside a recorded run", file=sys.stderr)
            return 2
        rest = [a for a in argv if a != "--record"]
        cmd = [sys.executable, "-m", "research", "run", "--tool", "check", "--project", "verity", "--cwd", "source", "--",
               sys.executable, "tools/check/check.py", *rest]
        return subprocess.run(cmd, cwd=ROOT).returncode
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
