"""`check`: the repository's one check, what a merge into `main` requires (`research merge`).

    uv run python tools/check/check.py                      run every step here, in the checkout
    uv run python tools/check/check.py --record [--on M]    the same through `research run --tool check` (on machine M, a
                                                            multi-core CPU pod): an Attempt of the clean commit checked out, the
                                                            record `research merge` reads; with the pinned upstream build and
                                                            inputs when the commit changes backends/flock/ against --base REF
                                                            (default origin/main)

Three independent groups run at once, the cores split between them (`--cores N`, default the CPUs this process may use):

    pytest          every test suite (`tools/check/suites.py`), half the cores' worth at once: each package's tests in their own
                    process, a pass reused while its inputs are unchanged; each suite's log and the outcomes (`suites.json`)
                    beside the run's result
    circuit-check   `circuit-check --all --jobs` the other half: every registered Definition and template (`tools/circuit_check`),
                    its report beside the run's result; then flock-circuit-build, `backends/flock/check_build.sh`: `cargo check` of
                    flock-circuit in every CPU feature set a pod builds, the proving builds without `seed-injection` among them (a
                    Flock b684b12 checkout, cached under $FLOCK_CHECKOUT)
    lean            `lake build` of the Lean verifier (`backends/flock/verifier/lean`), then its partition check against
                    `verity.ir.partition` (`unit_cut_agree.py`), then the audit of every Lake package with a `lean-audit.json`
                    (`tools/lean/audit.py --all --build`: axioms, sorry, declared axioms, escapes, files, kernel bypasses, code
                    run while compiling, dependency .olean digests, pinned statements, a kernel replay; its reports under
                    `lean-audit/`), then its agreement with upstream's Rust verifier (`ci.py`) on the upstream build stored once
                    and pinned by hash (`backends/flock/verifier/upstream.json`), when the run has it; skipped by name without
                    it.  `research merge` requires it for a change under backends/flock/ (`tool.py`)

Caches (``$CHECK_CACHE``, default ``~/.cache/verity-check``; ``--no-cache`` for a cold run), each a hit only on identical inputs:

    pytest          `suites.py`'s own: a suite's pass is reused while its declared inputs are unchanged (`--no-cache` passes
                    `--fresh`)
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
#: the Lean verifier of record, and `ci.py`, its agreement job against upstream's Rust verifier
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
        try:                                           # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            if q > 0:
                n = min(n, max(1, q // int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())))
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
    """The Lean steps: nothing on a tree without the verifier (a skip, named); then `lake build`, the Lean partition check's
    conformance with `verity.ir.partition` (`unit_cut_agree.py`), the audit of every Lean package, and the agreement job
    against upstream's Rust verifier on the pinned upstream build (`upstream.json`) when the run has it; without it that step
    is skipped by name.  `research merge` requires it for a change under `backends/flock/` (`tools/check/tool.py`), and
    `--record` sends it then."""
    lean = FLOCK_VERIFIER / "lean"
    if not (lean / "lakefile.toml").exists():
        return [("lean", "skipped: backends/flock/verifier/lean is not on this tree", ROOT)]
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
            ("lean-audit", [sys.executable, str(ROOT / "tools" / "lean" / "audit.py"), "--all", "--build", "--root", str(ROOT),
                            "--out", str(out / "lean-audit")], ROOT),
            ("lean-agreement", agreement, ROOT)]


def memory_gb() -> int:
    """The memory this process may use: the cgroup limit when there is one, v2 or v1 (a pod's /proc/meminfo is the host's),
    else MemAvailable."""
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):     # cgroup v2, then v1
        try:
            m = Path(f).read_text().strip()
            if m != "max" and int(m) < 1 << 60:        # v1 writes "no limit" as a number near 2^63
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


#: the agreement's memory floor: set 13's upstream replay (GEMM, m = 26) alone takes more than 15 GB
AGREEMENT_MIN_GB = 24


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def agreement_preflight(pin: dict, bins: list[Path]) -> str | None:
    """Why this machine cannot run the agreement (None when it can): too little memory, a glibc older than the pinned build
    needs, or a binary the loader refuses.  Seconds, where the job itself would take an hour to disagree on every session."""
    if (gb := memory_gb()) < AGREEMENT_MIN_GB:
        return (f"this machine may use {gb} GB; the agreement needs {AGREEMENT_MIN_GB} (set 13's upstream replay alone takes more "
                f"than 15 GB): record check on a larger machine")
    need, have = pin["upstream"].get("glibc"), os.confstr("CS_GNU_LIBC_VERSION") or ""
    if need and _version(have) < _version(need):
        return (f"the pinned upstream build needs glibc {need}; this machine has {have or 'none'}: record check on a newer "
                f"machine, or rebuild on this glibc (backends/flock/verifier/upstream-build.sh) and re-pin")
    for b in bins:
        try:
            r = subprocess.run([str(b)], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        except OSError as e:                                  # not an executable for this machine at all
            return f"{b.name} does not start here: {e}"
        err = r.stderr + r.stdout
        if "error while loading shared libraries" in err or re.search(r"version `GLIBC_[\d.]+' not found", err):
            return f"{b.name} does not start here: {err.strip().splitlines()[0]}"
    return None


def lean_agreement(out: Path, upstream: Path, inputs: Path) -> int:
    """Unpack the pinned upstream build and inputs, check every binary against the pin and that this machine can run them, and
    run `ci.py` on every replayable set."""
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
    if why := agreement_preflight(pin, [w / "upstream" / f"upstream-{c}" for c in sorted(pin["upstream"]["binaries"])]):
        print(f"check: lean-agreement cannot run here: {why}", file=sys.stderr)
        return 1
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
    from research.merge import rule_touches

    need = [s for s, rule in tool.CHECK.merge_requires.items() if rule_touches(rule, changed)]
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
    # vLLM's suite is not in the gate yet: 12 of its 4,400 CPU tests fail with torch 2.14 and it takes 20 minutes on one core
    commit = ["--commit", os.environ["RESEARCH_SOURCE_SHA"]] if os.environ.get("RESEARCH_SOURCE_SHA") else []   # a shipped tree has no .git
    pytest = [py, str(ROOT / "tools" / "check" / "suites.py"), "--json", str(out / "suites.json"), "--logs", str(out / "suites"),
              "--skip", "verity-vllm", *commit, "--jobs", str(a), *([] if cache else ["--fresh"])]
    cc = [py, "-m", "circuit_check", "--all", "-q", "--jobs", str(b), "--out", str(out / "circuit-check.json"),
          *(["--cache", str(cache_dir() / "circuit-check")] if cache else [])]
    build = ["bash", str(ROOT / "backends" / "flock" / "check_build.sh")]
    return [[("pytest", pytest, ROOT)], [("circuit-check", cc, ROOT), ("flock-circuit-build", build, ROOT)], lean_steps(out)]


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


def _env_key(e: dict[str, str]) -> dict[str, str]:
    """The environment as a key: without the shell's bookkeeping, and with this tree's path written `<tree>` (a machine keeps each
    commit's tree at its own path, and `uv run` puts that path in PATH, VIRTUAL_ENV and PYTHONPATH)."""
    return {k: v.replace(str(ROOT), "<tree>") for k, v in sorted(e.items()) if k not in VOLATILE}


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
    key = agreement_key(e) if cache and name == "lean-agreement" else None
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


def agreement_verdicts(run_dir: Path) -> list[str]:
    """The agreement's small outputs, published with the run: its result and each set's verdict table and verdict logs."""
    w = run_dir / "flock-agreement"
    files = [w / "result.json", *sorted(w.glob("work/set*/agree/agreement.tsv")), *sorted(w.glob("work/set*/agree/lean.txt")),
             *sorted(w.glob("work/set*/agree/upstream.txt"))]
    return [str(p.relative_to(run_dir)) for p in files if p.is_file()]


def write_result(run_dir: Path, ok: bool, done: list[dict]) -> None:
    doc = {"schema": "research/result/v0.1", "kind": "check/v1", "run_id": os.environ["RESEARCH_RUN_ID"],
           "workload_fingerprint": {"steps": [s["step"] for s in done]},
           "validation": {"status": "passed" if ok else "failed", "checker": "tools/check/check.py",
                          "detail": "; ".join(f"{s['step']} {s['status']}" for s in done)},
           "measurements": [{"name": f"{s['step']}_wall", "value": s["seconds"], "unit": "s"} for s in done],
           "steps": done, "artifacts": [p for p in ("suites.json", "suites/", "circuit-check.json", "lean-audit/",
                                                    *(s.get("log") for s in done), *agreement_verdicts(run_dir))
                                        if p and (run_dir / p).exists()]}
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
