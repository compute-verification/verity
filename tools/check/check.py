"""`check`: the repository's one check, what a merge into `main` requires (`research merge`).

    uv run --extra torch-cpu python tools/check/check.py    run every step here, in the checkout, with CPU-only torch
    ... [--keep-going]                                      run every step to its end, even after one has failed
    uv run python tools/check/check.py --record [--on M]    the same through `research run --tool check` (on machine M, a
                                                            multi-core CPU pod): an Attempt of the clean commit checked out, the
                                                            record `research merge` reads; with the pinned upstream build and
                                                            inputs when the commit changes backends/flock/ against --base REF
                                                            (default origin/main), and presigned URLs for the pinned Lean
                                                            dependency bundles; with --on, in a free check slot of M
                                                            (`tools/check/slot.py`), the lock and cores the train checks take;
                                                            --priority takes the next free slot ahead of waiting lane checks;
                                                            --agreement sends the upstream build whatever the commit changes
                                                            (a train that changes a pinned guarantee elsewhere: `train.sh`);
                                                            a warning first when the commit can't land on REF as it is
    uv run python tools/check/check.py --lean-deps-files    the `--send` for those URLs, for `research merge --train`
    python tools/check/check.py --preflight                 the preflight alone, for `research merge --train --prepare`
    ... [--verdicts-in PACK]                                import the passes another machine stored (`verdicts.py`) before
                                                            the steps; with --record --on, the pack travels with the run

First the preflight, seconds in all: whether `uv.lock` is current (`uv lock --check`), the repository's wall-clock lint and its
tree's own tests (`LINTS`: markdown caps, report-genre names, secret-like literals, blob and data budgets). The groups start only
once it passes, so a check that one of them fails ends in seconds, not hours.

A step that fails stops the others: each running step's processes get SIGINT (Python code runs its `finally`s, so the Lean audit
removes its scratch tree) and whatever is left of them SIGKILL after `STOP_GRACE_S`, and no group starts another step. Such a
step's status is `stopped`. `--keep-going` lets every group run to its end instead, keeping the verdicts its passing steps store,
except after a step in `HALTS`: a Lean module that doesn't compile fails the audit too, so nothing still running is worth the slot.
The steps stay in this process's group, which a runner's timeout and a manual stop signal as a whole. Each step's record is
appended to `steps.jsonl` in the output directory as the step ends, so a failure shows while the rest still run.

Then the independent groups run at once, the cores split between the first two (`--cores N`, default the CPUs this process may
use):

    pytest          every test suite (`tools/check/suites.py`) but those that use Lean, half the cores' worth at once: each
                    package's tests in their own process, a pass reused while its inputs are unchanged; each suite's log and the
                    outcomes (`suites.json`) beside the run's result
    circuit-check   `circuit-check --all --jobs` the other half: every registered Definition and template (`tools/circuit_check`),
                    its report beside the run's result; then flock-circuit-build, `backends/flock/check_build.sh`: `cargo check` of
                    flock-circuit in every CPU feature set a pod builds, the proving builds without `seed-injection` among them (a
                    Flock b684b12 checkout, cached under $FLOCK_CHECKOUT); then flock-rust-tests, `check_build.sh test` on that
                    checkout: `cargo test --release` of flock-live's unit tests and of the five upstream crates the patches change
    lean            `lake build` of the Lean verifier (`backends/flock/verifier/lean`), then its partition check against
                    `verity.ir.partition` (`unit_cut_agree.py`), then the audit of every Lake package with a `lean-audit.json`
                    (`tools/lean/audit.py --all --build`: axioms, sorry, declared axioms, escapes, files, kernel bypasses, code
                    run while compiling, dependency .olean digests, the pins' records, a kernel replay; its reports under
                    `lean-audit/`), then the suites that use Lean (`AFTER_LEAN`, outcomes in `suites-lean.json`) on what those
                    steps built, then its agreement with upstream's Rust verifier (`ci.py`) on the upstream build stored once
                    and pinned by hash (`backends/flock/verifier/upstream.json`), when the run has it; skipped by name without
                    it.  `research merge` requires it for a change under backends/flock/ (`tool.py`)
    lean-changed    at once with the lean group: `lake build` of each Lake package whose audit misses, on a tree this machine
                    keeps built (`lean_changed.py`), so a module that no longer compiles ends the run in minutes, not when the
                    audit's cold build reaches it; it skips what it can't build here and vouches for nothing: the audit decides

Caches (``$CHECK_CACHE``, default ``~/.cache/verity-check``; ``--no-cache`` for a cold run), each a hit only on identical inputs:

    pytest          `suites.py`'s own: a suite's pass is reused while its declared inputs are unchanged (`--no-cache` passes
                    `--fresh`)
    circuit-check   `circuit-check --cache`, per target: a target's result is reused when every file, directory listing,
                    existence probe and loaded library its process read is unchanged, with the checker's own version; one after
                    a read the tracer cannot see (a program it starts) is not stored (`circuit_check.cache`)
    lean-audit      per package (`lean_audit.py`): a pass is reused while the package, its path dependencies, `tools/lean/`
                    and its toolchain and manifest are unchanged; a miss is audited on a scratch tree holding only those, so a
                    read outside them fails there; fetched dependencies stay warm on the machine under manifest + toolchain,
                    and a machine without them restores the bundle `lean-deps.json` pins from the store, not from GitHub; a
                    step that audits any package first waits for one of the node's Lean audit slots (`lean_slot.py`)
    lean-changed    not a verdict: its tree under `lean-changed/` keeps each package's last build there, so Lake rebuilds only
                    what changed and what imports it, and its fetched dependencies are hard links to lean-audit's, read-only
    lean-agreement  a pass is reused when everything the job reads is the same (`agreement_key`: backends/flock/verifier/'s
                    tracked files, which pin the upstream build and inputs, the interpreter, the environment)
    flock-rust-tests  likewise (`rust_tests_key`: the script, which pins the Flock commit and the Rust toolchain, the patches,
                    the live crate, that toolchain's `rustc -vV`, the environment), so only a change to those pays for the Rust
                    tests

Inside a recorded run it writes `$RESEARCH_RUN_DIR/result.json` (kind `check/v1`): each step's status, seconds and cache use,
with each step's log beside it, and `verdicts.tar.gz`: the passes this run stored, which another machine imports with
`--verdicts-in` (the merge queue's pods share their passes this way; the pack imported is named in the result's `verdicts`).

The evidence store.  A recorded check runs the store-backed tests (integrations/vllm's `store_io`) on the store's remote, never
skipped: its steps get VERITY_STORE_REQUIRED=1 (a missing remote fails those tests) and, as CHECK_STORE_CUSTODY, the run's custody
dir (`research run --custody-r2`, the default with --on: a delete-free key with GetObject), which `store_io` hands to its store
CLI only.  So `--record --on` must be launched where the parent R2 key is (the Cursor secrets, or `~/.config/verity/r2.env` when it
holds no AWS_SESSION_TOKEN): a temporary credential cannot mint the custody key, and `research run` refuses.  VERITY_SKIP_STORE=1
is refused inside a recorded run.  Outside one, the store-backed tests skip where no remote is configured.  Under
VERITY_STORE_REQUIRED=1 the suites' guard fails any skip whose reason says "the evidence store is not reachable", whichever helper
fetched (C-Flock's `verifier/ci.py` hands the custody key to its store CLI too).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: the Lean verifier of record, and `ci.py`, its agreement job against upstream's Rust verifier
FLOCK_VERIFIER = ROOT / "backends" / "flock" / "verifier"
#: environment variables that say nothing about what a step computes (the shell's own bookkeeping)
VOLATILE = frozenset({"_", "OLDPWD", "PWD", "SHLVL", "TERM", "COLUMNS", "LINES", "TMUX", "TMUX_PANE", "SSH_CLIENT", "SSH_CONNECTION",
                      "SSH_TTY", "SSH_AUTH_SOCK", "XDG_SESSION_ID", "MAIL", "LOGNAME", "USER", "HOSTNAME", "PUBLIC_KEY",
                      "CHECK_STORE_CUSTODY"})
#: variables naming the machine rather than what a step computes: a pod's id, address and pod-scoped key (so a verdict one pod
#: stored stays valid on another; `verdicts.py`)
MACHINE_PREFIXES = ("RUNPOD_",)
#: the recording run's custody dir, for `store_io` (not VERITY_-prefixed: the suites' cache keys on those, and the path is per run)
CUSTODY_ENV = "CHECK_STORE_CUSTODY"
STORE_REQUIRED_ENV = "VERITY_STORE_REQUIRED"


def custody_dir() -> Path | None:
    """The recording run's custody dir (`<root>/requests/<run id>/custody/`, beside `<root>/runs/<run id>/`) when it holds a key."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return None
    d = Path(run_dir).resolve().parent.parent / "requests" / run_id / "custody"
    return d if (d / "store.toml").is_file() and (d / "cred.json").is_file() else None


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"check_{name}", Path(__file__).resolve().parent / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


#: the verdict packs (`verdicts.py`)
VERDICTS = _load("verdicts")
#: the helpers `lean_audit.py` shares (`common.py`), kept out of this file so that its audit key doesn't cover this file
COMMON = _load("common")
cache_dir = COMMON.cache_dir
cores = COMMON.cores
_sha256 = COMMON.sha256
tracked_files = COMMON.tracked_files


#: the toolchain `lean-toolchain` pins, through elan when `lake` is missing (as `ci-pod.sh` does)
ELAN = ('command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || { curl -sSf '
        'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }; '
        'export PATH="$HOME/.elan/bin:$PATH"; ')


#: upstream's verifier builds and the sets' stored inputs, stored once and pinned by hash (`upstream-build.sh`)
UPSTREAM_PIN = FLOCK_VERIFIER / "upstream.json"
#: where check looks for them besides the run's `inputs/` (`--send`): a local recorded run's launcher sets it
AGREEMENT_DIR_ENV = "FLOCK_AGREEMENT_INPUTS"


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


#: the presigned URLs of the pinned Lean dependency bundles (`lean_audit.py`), among a recorded run's inputs, or named by this
#: variable for a local one
LEAN_DEPS_URLS = "lean-deps-urls.json"
LEAN_DEPS_URLS_ENV = "CHECK_LEAN_DEPS_URLS"
#: what a `--record --on` check runs under on the machine: a free check slot's lock and cores (slot.py), from the shipped tree
SLOT_WRAP = ["python3", "tools/check/slot.py", "--"]


def slot_wrap(priority: bool = False) -> list[str]:
    """SLOT_WRAP naming who asked (RESEARCH_LANE, else RESEARCH_AGENT_URL), which `slot.py --status` on the machine shows, and
    with --priority for the lander's train checks."""
    by = os.environ.get("RESEARCH_LANE") or os.environ.get("RESEARCH_AGENT_URL")
    return [*SLOT_WRAP[:-1], *(["--priority"] if priority else []), *(["--by", by] if by else []), SLOT_WRAP[-1]]


def take_lean_deps_urls(out: Path) -> Path | None:
    """The presigned URLs sent with the run, moved out of the run directory before any step starts: each grants a read until it
    expires, and the run's record keeps every file left there."""
    sent = out / "inputs" / LEAN_DEPS_URLS
    if not sent.is_file():
        return Path(os.environ[LEAN_DEPS_URLS_ENV]) if os.environ.get(LEAN_DEPS_URLS_ENV) else None
    fd, tmp = tempfile.mkstemp(prefix="lean-deps-urls-", suffix=".json")
    with os.fdopen(fd, "wb") as f:
        f.write(sent.read_bytes())
    sent.unlink()
    return Path(tmp)


def _lean_audit():
    spec = importlib.util.spec_from_file_location("check_lean_audit_urls", ROOT / "tools" / "check" / "lean_audit.py")
    la = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(la)
    return la


def lean_deps_url_file(dest: Path) -> Path | None:
    """`lean-deps-urls.json` for this tree's pinned Lean dependency bundles, signed by this machine's store remote."""
    return _lean_audit().url_file(dest)


#: the lean-audit step's wrapper when it audits anything: one of the node's Lean audit slots, from pool `check` (lean_slot.py)
LEAN_SLOT_WRAP = [sys.executable, str(ROOT / "tools" / "check" / "lean_slot.py"), "--pool", "check", "--"]


def lean_audit_misses(cache: bool = True) -> bool:
    """Whether the lean-audit step will audit a package: one whose pass isn't kept here under its key (`lean_audit.run`'s
    test)."""
    la = _lean_audit()
    files = tracked_files(ROOT)
    if not cache or files is None:
        return True
    return any(not (cache_dir() / "lean-audit" / f"{la.key(p, files, la.FLAGS)}.json").is_file() for p in la.packages(files))


def lean_slot(cmd: list[str]) -> list[str]:
    """`--lean-slot CMD...`, the lean-audit step: CMD in one of the node's Lean audit slots when it will audit a package, as it
    is when it only reuses passes, which would otherwise wait behind an hour-long audit for seconds of work."""
    return [*LEAN_SLOT_WRAP, *cmd] if lean_audit_misses("--fresh" not in cmd) else cmd


def lean_steps(out: Path, cache: bool = True, urls: Path | None = None) -> list[tuple[str, list[str] | str, Path]]:
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
            ("lean-audit", [sys.executable, str(Path(__file__).resolve()), "--lean-slot",
                            sys.executable, str(ROOT / "tools" / "check" / "lean_audit.py"), "--out", str(out / "lean-audit"),
                            "--cache", str(cache_dir() / "lean-audit"), "--deps", str(cache_dir() / "lean-deps"),
                            *([] if cache else ["--fresh"]), *(["--urls", str(urls)] if urls else [])], ROOT),
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


def upstream_smoke(w: Path) -> list[tuple[str, list[str], dict]]:
    """One real replay per upstream binary that `vectors.json` gives a stored honest session (a set without lookup tables):
    (its PR #83 commit, the command `agree.py` would run, the verdict expected), over the unpacked build and inputs in ``w``."""
    import argparse

    spec = importlib.util.spec_from_file_location("check_flock_agree", FLOCK_VERIFIER / "agree.py")
    agree = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agree)

    def fetch(art: str) -> Path:                        # ci.py's layout: art:<id>[/path] under inputs/<id>/
        ref, _, path = art.partition("/")
        d = w / "inputs" / ref.split(":", 1)[1]
        return d / path if path else sorted(d.iterdir())[0]

    out: dict[str, tuple] = {}
    for s in json.loads((FLOCK_VERIFIER / "vectors.json").read_text())["replayable"]["sets"]:
        st = s.get("statement", {})
        if s["pr83"] in out or not s.get("sessions") or "tables" in st:
            continue
        honest = fetch(s["sessions"][0]["dir"])
        a = argparse.Namespace(upstream=str(w / "upstream" / f"upstream-{s['pr83']}"), statement=s["lean_statement"],
                               circuit=str(fetch(st["circuit"])), tables=None)
        public = honest / "pub.bin" if (honest / "pub.bin").exists() else fetch(st["public"])
        extra = ["--drawn-only"] if agree.drawn(str(honest)) else []
        out[s["pr83"]] = (s["pr83"], [*agree.upstream_cmd(a, str(public), extra), str(honest)], s["sessions"][0]["expect"])
    return list(out.values())


def agreement_preflight(pin: dict, bins: list[Path], w: Path | None = None) -> str | None:
    """Why this machine cannot run the agreement (None when it can): too little memory, a glibc older than the pinned build
    needs, a binary the loader refuses, or (with ``w``, the unpacked build and inputs) one that cannot replay a stored honest
    session here: a build for another CPU dies of SIGILL on the first instruction this one lacks, and prints nothing.
    Seconds, where the job itself would take an hour to disagree on every session."""
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
    if w is None:
        return None
    smoke = upstream_smoke(w)
    if not smoke:
        return "vectors.json gives no stored honest session to replay, so nothing shows the upstream build runs here"
    flags = pin["upstream"].get("rustflags", "the build host's own (target-cpu=native)")
    for c, cmd, expect in smoke:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return f"upstream-{c} did not finish replaying its stored honest session in 600 s"
        if r.returncode < 0:
            sig = signal.Signals(-r.returncode).name
            return (f"upstream-{c} was killed by {sig} replaying its stored honest session"
                    + (f": this CPU lacks an instruction the build uses (target flags: {flags}); rebuild for a CPU every check "
                       f"machine has (upstream-build.sh) and re-pin" if sig == "SIGILL" else ""))
        got = [json.loads(line.split("\t", 1)[1])["verdict"] for line in r.stdout.splitlines() if line.startswith("UPSTREAM\t")]
        if r.returncode or not got or got[0]["accepted"] != expect["accepted"]:
            return (f"upstream-{c} replayed its stored honest session with exit {r.returncode} and verdict "
                    f"{got[0] if got else 'none'} (expected {expect}): {(r.stderr or r.stdout).strip()[-300:]}")
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
    if why := agreement_preflight(pin, [w / "upstream" / f"upstream-{c}" for c in sorted(pin["upstream"]["binaries"])], w):
        print(f"check: lean-agreement cannot run here: {why}", file=sys.stderr)
        return 1
    ci = pin.get("ci", {})
    sets = json.loads((FLOCK_VERIFIER / "vectors.json").read_text())["replayable"]["sets"]    # ci.py --jobs: sets at once
    argv = [sys.executable, str(FLOCK_VERIFIER / "ci.py"), "--lean", str(FLOCK_VERIFIER / "lean" / ".lake" / "build" / "bin" / "flock-verify"),
            *ups, "--work", str(w / "work"), "--inputs", str(w / "inputs"), "--jobs", str(max(1, min(cores(), len(sets)))),
            "--session-jobs", str(session_jobs()),
            "--seed", str(ci.get("seed", 20260926)), "--fuzz", str(ci.get("fuzz", 20)), "--out", str(w)]
    return subprocess.run(argv, cwd=ROOT).returncode


def landable(base: str) -> bool:
    """Whether `research merge` could land HEAD on ``base`` as it is: ``base`` is in HEAD, or merging HEAD into it leaves HEAD's own
    tree (the gate's `tree_identical`). Otherwise the tree that lands is a merge nobody checked yet."""
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT, capture_output=True).returncode == 0:
        return True
    r = subprocess.run(["git", "merge-tree", "--write-tree", base, "HEAD"], cwd=ROOT, capture_output=True, text=True)
    head = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == head


def record(argv: list[str]) -> int:
    """`--record [--on MACHINE] [--base REF] [--agreement]`: `research run --tool check` on this clean commit, with the pinned
    upstream build and inputs (fetched from the store) when the commit changes a path `tools/check/tool.py` names against REF
    (default origin/main, fetched first), or whenever --agreement is given. A commit that can't land on REF as it is
    (`landable`) gets a warning and is checked anyway."""
    on = base = None
    agreement = priority = False
    rest = []
    it = iter(argv)
    for x in it:
        if x == "--on":
            on = next(it)
        elif x == "--base":
            base = next(it)
        elif x == "--agreement":
            agreement = True
        elif x == "--priority":
            priority = True
        else:
            rest.append(x)
    spec = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    if base is None:
        subprocess.run(["git", "fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main"], cwd=ROOT, capture_output=True)
        base = "origin/main"
    if not landable(base):
        print(f"check: this commit can't land as is; the train's check is the merge check (merging it into {base} gives a tree "
              f"other than its own, so `research merge` won't take this check)")
    changed = [p for p in subprocess.run(["git", "diff", "--name-only", "-z", base, "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                         errors="surrogateescape", check=True).stdout.split("\0") if p]
    from research.merge import rule_touches

    need = [s for s, rule in tool.CHECK.merge_requires.items() if rule_touches(rule, changed)] + ["lean-agreement"] * agreement
    cmd = [sys.executable, "-m", "research", "run", "--tool", "check", "--project", "verity", "--cwd", "source"]
    env = dict(os.environ)
    if "lean-agreement" in need:
        from research.store.local import LocalStore

        pin = json.loads(UPSTREAM_PIN.read_text())
        store = LocalStore()
        files = {k: Path(store.fetch(pin[k]["art"])) for k in ("upstream", "inputs")}
        print(f"check: changes under {', '.join(r for r in tool.CHECK.merge_requires['lean-agreement'] if not r.startswith('!'))} "
              f"against {base}: sending the pinned "
              f"upstream build and inputs")
        cmd += [x for k in ("upstream", "inputs") for x in ("--input", f"flock_{k}={pin[k]['art']}")]
        if on:
            cmd += [x for f in files.values() for x in ("--send", str(f))]
        else:
            d = Path(tempfile.mkdtemp(prefix="check-agreement-"))
            for k, f in files.items():
                (d / f"{k}.tar.gz").symlink_to(f)
            env[AGREEMENT_DIR_ENV] = str(d)
    d = Path(tempfile.mkdtemp(prefix="check-lean-deps-"))
    if urls := lean_deps_url_file(d):
        print(f"check: sending presigned URLs for {len(json.loads(urls.read_text()))} pinned Lean dependency bundle(s)")
        if on:
            cmd += ["--send", str(urls)]
        else:
            env[LEAN_DEPS_URLS_ENV] = str(urls)
    if on:
        cmd += ["--on", on, "--source", "."]
        if "--verdicts-in" in rest:                    # the pack travels with the run; on the machine it's found in the run's inputs/
            cmd += ["--send", str(Path(rest[rest.index("--verdicts-in") + 1]).resolve())]
    cmd += ["--", *(slot_wrap(priority) if on else []), *tool.CHECK.command, *rest]
    try:
        return subprocess.run(cmd, cwd=ROOT, env=env).returncode
    finally:
        shutil.rmtree(d, ignore_errors=True)


def verdicts_in(pack: Path, run_dir: Path | None) -> dict:
    """Import a verdict pack before the steps run (`verdicts.py`): the path given, else its name in a recorded run's `inputs/`."""
    if not pack.exists() and run_dir is not None and (run_dir / "inputs" / pack.name).exists():
        pack = run_dir / "inputs" / pack.name
    added, kept = VERDICTS.import_(pack)
    print(f"check: verdicts: {added} imported, {kept} already here, from {pack.name}", flush=True)
    return {"pack": pack.name, "sha256": _sha256(pack), "added": added, "already": kept}


#: The suites whose tests run `lake` or what the Lean steps build under `.lake/`: a step of the Lean group once `lean-audit` has
#: built every Lake package, with elan's `lake` on the PATH, so what they run or skip never depends on how far a concurrent
#: `lake build` has got, and no test's `lake build` shares a build directory with one (tests/test_no_wall_clock.py holds every
#: other suite to not skipping on them).
AFTER_LEAN = ("verity-flock", "verity-lean-audit")


def groups(out: Path, n: int, cache: bool = True, urls: Path | None = None) -> list[list[tuple[str, list[str] | str, Path]]]:
    """The independent groups, each a sequence of (name, argv or the reason it is skipped / cannot run, cwd)."""
    py = sys.executable
    a = max(1, n // 2)
    b = max(1, n - a)
    suites = [py, str(ROOT / "tools" / "check" / "suites.py")]
    tail = ["--jobs", str(a), *([] if cache else ["--fresh"])]
    lean = lean_steps(out, cache, urls)
    after_lean: list[str] = []
    if len(lean) > 1:
        after_lean = [x for s in AFTER_LEAN for x in ("--skip", s)]
        lean_suites = [*suites, *AFTER_LEAN, "--json", str(out / "suites-lean.json"), "--logs", str(out / "suites"), *tail]
        lean.insert(3, ("lean-suites", ["bash", "-c", ELAN + shlex.join(lean_suites)], ROOT))
    pytest = [*suites, "--json", str(out / "suites.json"), "--logs", str(out / "suites"), *after_lean, *tail]
    cc = [py, "-m", "circuit_check", "--all", "-q", "--jobs", str(b), "--out", str(out / "circuit-check.json"),
          *(["--cache", str(cache_dir() / "circuit-check")] if cache else [])]
    build = ["bash", str(ROOT / "backends" / "flock" / "check_build.sh")]
    rust = ["env", f"CARGO_BUILD_JOBS={b}", f"RUST_TEST_THREADS={b}", *build, "test"]
    gs = [[("pytest", pytest, ROOT)], [("circuit-check", cc, ROOT), ("flock-circuit-build", build, ROOT), ("flock-rust-tests", rust, ROOT)],
          lean]
    if len(lean) > 1:
        changed: list[str] | str = ([py, str(ROOT / "tools" / "check" / "lean_changed.py"), "--cache", str(cache_dir() / "lean-audit"),
                                     "--deps", str(cache_dir() / "lean-deps"), "--tree", str(cache_dir() / "lean-changed")]
                                    if cache else "skipped: --no-cache: lean-audit builds every package cold")
        gs.append([("lean-changed", changed, ROOT)])
    return gs


def steps(out: Path, n: int = 1) -> list[tuple[str, list[str] | str, Path]]:
    return [s for g in groups(out, n) for s in g]


#: What reaches the steps from this process's environment, Turborepo's strict mode being the model: where things are (PATH, HOME,
#: the temporary and XDG directories, the Rust and Lean toolchains' homes), the locale, how this machine reaches the network (its
#: proxies and certificate bundles), uv and the environment it runs in, and the switches tests read (`VERITY_*`, `FLOCK_*`).
#: Nothing else: not the store's, RunPod's or GitHub's credentials, not what the recording run gives its workload (`RESEARCH_*`,
#: the machine's PYTHONPYCACHEPREFIX; the tests start runs of their own, which would inherit them).
ENV_PASS = frozenset({"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LANGUAGE", "VIRTUAL_ENV", "CARGO_HOME", "RUSTUP_HOME",
                      "ELAN_HOME", "SSL_CERT_FILE", "SSL_CERT_DIR", "http_proxy", "https_proxy", "no_proxy", "all_proxy",
                      "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY"})
ENV_PASS_PREFIXES = ("LC_", "XDG_", "UV_", "VERITY_", "FLOCK_")
#: What check sets for its steps from its own state, never from the environment: a check run inside a step (its own tests) would
#: otherwise pass its parent's on.
ENV_OWN = frozenset({CUSTODY_ENV, STORE_REQUIRED_ENV})


def env() -> dict[str, str]:
    """The steps' environment: this process's variables that `ENV_PASS` and `ENV_PASS_PREFIXES` name, less `ENV_OWN`; PYTHONPATH
    entries inside this tree only (a machine's runner puts its shipped `research` first); and a git identity.  Inside a recorded
    run, the store is required and the run's custody dir is passed on (the module docstring's "The evidence store")."""
    out = {k: v for k, v in os.environ.items() if (k in ENV_PASS or k.startswith(ENV_PASS_PREFIXES)) and k not in ENV_OWN}
    if os.environ.get("RESEARCH_RUN_ID"):
        out[STORE_REQUIRED_ENV] = "1"
        if custody := custody_dir():
            out[CUSTODY_ENV] = str(custody)
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


def _env_key(e: dict[str, str]) -> dict[str, str]:
    """The environment as a key: without the shell's bookkeeping or the machine's identity, and with this tree's path written `<tree>`
    (a machine keeps each commit's tree at its own path, and `uv run` puts that path in PATH, VIRTUAL_ENV and PYTHONPATH)."""
    return {k: v.replace(str(ROOT), "<tree>") for k, v in sorted(e.items()) if k not in VOLATILE and not k.startswith(MACHINE_PREFIXES)}


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


def rust_tests_inputs(files: list[str]) -> list[str]:
    """What `check_build.sh test` reads of the tree: the script (which pins the Flock commit and the order the patches go on in),
    the patches, and the live crate it links in."""
    pre = "backends/flock/"
    return [f for f in files if f == pre + "check_build.sh" or f.startswith(pre + "live/")
            or (f.startswith(pre + "flock-") and f.endswith("-b684b12.patch"))]


def rust_toolchain() -> str | None:
    """The Rust toolchain `check_build.sh` pins: its `RUST_TOOLCHAIN` default (check passes no `RUST_TOOLCHAIN` of its own)."""
    p = ROOT / "backends" / "flock" / "check_build.sh"
    m = re.search(r"^RUST_TOOLCHAIN=\$\{RUST_TOOLCHAIN:-([^}]+)\}$", p.read_text(), re.M) if p.is_file() else None
    return m and m.group(1)


def rust_tests_key(e: dict[str, str]) -> str | None:
    """What flock-rust-tests reads: `rust_tests_inputs`, the Rust toolchain the script pins (`rustc -vV` under
    `RUSTUP_TOOLCHAIN`, ~/.cargo/bin first), the platform and the environment.  None when the tree's files cannot be listed, the
    script pins no toolchain, or that toolchain isn't installed yet (the script installs it; that run stores nothing)."""
    files, toolchain = tracked_files(), rust_toolchain()
    path = f"{Path(e.get('HOME') or Path.home()) / '.cargo' / 'bin'}{os.pathsep}{e.get('PATH', '')}"
    rustc = shutil.which("rustc", path=path)
    if files is None or toolchain is None or rustc is None:
        return None
    r = subprocess.run([rustc, "-vV"], env={**e, "PATH": path, "RUSTUP_TOOLCHAIN": toolchain}, capture_output=True, text=True)
    if r.returncode:
        return None
    h = hashlib.sha256(json.dumps([r.stdout, platform.platform(), _env_key(e)]).encode())
    for f in rust_tests_inputs(files):
        p = ROOT / f
        h.update(f.encode() + b"\0" + (_sha256(p).encode() if p.is_file() else b"-") + b"\0")
    return h.hexdigest()


#: the steps whose pass is kept under a key of everything they read, and reused while it holds
STEP_KEYS = {"lean-agreement": agreement_key, "flock-rust-tests": rust_tests_key}


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


#: the repository-wide lints the preflight runs: each fails a whole check, and together they take seconds
LINTS = ("tests/test_no_wall_clock.py", "tests/test_repository.py")
#: how long a stopped step's processes have after SIGINT before SIGKILL
STOP_GRACE_S = 60
#: each step's record, one JSON line appended as the step ends
STEPS_LOG = "steps.jsonl"


def preflight() -> list[tuple[str, list[str] | str, Path]]:
    """The steps before the groups: `uv lock --check` (a current lock needs no index), then `LINTS` (without bytecode or
    pytest's cache, so they leave nothing in the tree). Under `uv run`, `$UV` names uv."""
    uv = os.environ.get("UV") or shutil.which("uv")
    lock = [uv, "lock", "--check"] if uv else "failed: no `uv` here to check `uv.lock` (run check through `uv run`)"
    return [("preflight-lock", lock, ROOT),
            ("preflight-lints", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *LINTS], ROOT)]


def preflight_only() -> int:
    """`--preflight`: every preflight step, its output here; 0 when all pass."""
    failed = []
    for name, argv, cwd in preflight():
        ok = not isinstance(argv, str) and subprocess.run(argv, cwd=cwd).returncode == 0
        print(f"check: {name}: {'passed' if ok else argv if isinstance(argv, str) else 'FAILED'}", flush=True)
        failed += [] if ok else [name]
    return 1 if failed else 0


def _tree(pid: int) -> list[int]:
    """``pid`` and its descendants, read from /proc (only ``pid`` where there is none)."""
    kids: dict[int, list[int]] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return [pid]
    for d in names:
        if d.isdigit():
            try:
                ppid = int(Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def _signal_tree(pid: int, sig: int) -> None:
    for p in _tree(pid):
        try:
            os.kill(p, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _kill_after_grace(p: subprocess.Popen) -> None:
    try:
        p.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        _signal_tree(p.pid, signal.SIGKILL)


#: the steps whose failure stops the run even under --keep-going
HALTS = frozenset({"lean-changed"})


class Steps:
    """The steps of one run: their processes while they run, so that the first failure stops the others (unless ``keep_going``,
    and a step in `HALTS` stops them even then), and each step's record appended to `STEPS_LOG` as it ends."""

    def __init__(self, out: Path, keep_going: bool = False):
        self.log, self.keep_going = out / STEPS_LOG, keep_going
        self.lock = threading.Lock()
        self.running: dict[str, subprocess.Popen] = {}
        self.stopped: set[str] = set()
        self.halted: str | None = None
        self.log.unlink(missing_ok=True)

    @property
    def stopping(self) -> bool:
        return self.halted is not None

    def start(self, name: str, argv: list[str], cwd: Path, env: dict[str, str], log) -> subprocess.Popen | None:
        """The step's process; None, and nothing started, once a failure has stopped the run."""
        with self.lock:
            if self.stopping:
                return None
            self.running[name] = p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            return p

    def end(self, rec: dict) -> dict:
        with self.lock:
            self.running.pop(rec["step"], None)
            if rec["step"] in self.stopped and rec["status"] != "passed":
                rec.update(status="stopped", detail=f"stopped: {self.halted} failed")
            rec["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(self.log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["status"] == "failed" and self.halted is None and (not self.keep_going or rec["step"] in HALTS):
                self.halted = rec["step"]
                for name, p in self.running.items():
                    self.stopped.add(name)
                    _signal_tree(p.pid, signal.SIGINT)
                    threading.Thread(target=_kill_after_grace, args=(p,), daemon=True).start()
        return rec


def _step(name: str, argv: list[str] | str, cwd: Path, out: Path, cache: bool, steps: Steps) -> dict:
    if isinstance(argv, str):
        skipped = argv.startswith("skipped:")
        print(f"check: {name}: {argv}", flush=True)
        return steps.end({"step": name, "status": "skipped" if skipped else "failed", "detail": argv, "seconds": 0.0})
    e = env()
    t0 = time.monotonic()
    hit = None
    key = STEP_KEYS[name](e) if cache and name in STEP_KEYS else None
    if key:
        entry = cache_dir() / name / f"{key}.json"
        if entry.exists():
            hit = key
    log = out / f"{name}.log"
    if hit:
        rec = {"step": name, "status": "passed", "rc": 0, "cache": {"hit": hit}}
    else:
        with open(log, "wb") as f:
            p = steps.start(name, argv, cwd, e, f)
            rc = p.wait() if p is not None else None
        if rc is None:
            rec = {"step": name, "status": "stopped", "detail": f"not started: {steps.halted} failed"}
        else:
            rec = {"step": name, "status": "passed" if rc == 0 else "failed", "rc": rc, "log": log.name}
        if rc == 0 and key:
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(json.dumps({"key": key, "argv": argv, "passed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            rec["cache"] = {"stored": key}
        if name == "circuit-check" and (out / "circuit-check.json").exists():
            rec["cache"] = json.loads((out / "circuit-check.json").read_text()).get("cache")
    rec["seconds"] = round(time.monotonic() - t0, 1)
    rec = steps.end(rec)
    if rec["status"] == "failed" and log.exists():
        tail = log.read_bytes().decode(errors="replace").splitlines()[-40:]
        print("\n".join(f"  {name}| {line}" for line in tail), flush=True)
    print(f"check: {name}: {rec['status']}{' (cached)' if hit else ''} in {rec['seconds']} s", flush=True)
    return rec


#: vLLM's C++ twin models, compiled on first use: built once before the groups start, so that no step compiles them while
#: another loads them (a check that compiles mid-run is untraceable, and circuit-check stores none of its results)
PREBUILD = ("try:\n    from verity_vllm.program.kernels import twins\nexcept ImportError:\n    twins = None\n"
            "if twins:\n    twins.preload()\n")


def run(out: Path, n: int | None = None, cache: bool = True, keep_going: bool = False) -> tuple[bool, list[dict]]:
    n = n or cores()
    urls = take_lean_deps_urls(out)
    steps = Steps(out, keep_going)

    def group(g: list) -> list[dict]:
        done = []
        for name, argv, cwd in g:
            done.append(_step(name, argv, cwd, out, cache, steps))
            if done[-1]["status"] in ("failed", "stopped"):
                break
        return done

    try:
        results = [_step(name, argv, cwd, out, cache, steps) for name, argv, cwd in preflight()]
        if not steps.stopping:
            subprocess.run([sys.executable, "-c", PREBUILD], cwd=ROOT, env=env(), capture_output=True)
            gs = groups(out, n, cache, urls)
            with ThreadPoolExecutor(max_workers=len(gs)) as ex:
                results += [r for rs in ex.map(group, gs) for r in rs]
    finally:
        if urls and urls.name.startswith("lean-deps-urls-"):
            urls.unlink(missing_ok=True)
    return not any(r["status"] in ("failed", "stopped") for r in results), results


def agreement_verdicts(run_dir: Path) -> list[str]:
    """The agreement's small outputs, published with the run: its result and each set's verdict table and verdict logs."""
    w = run_dir / "flock-agreement"
    files = [w / "result.json", *sorted(w.glob("work/set*/agree/agreement.tsv")), *sorted(w.glob("work/set*/agree/lean.txt")),
             *sorted(w.glob("work/set*/agree/upstream.txt"))]
    return [str(p.relative_to(run_dir)) for p in files if p.is_file()]


def contradictions(out: Path, field: str = "contradicts") -> list[dict]:
    """What the suites' runs say about the caches (`suites.py`), per suite: ``contradicts``, what a fresh run (`--no-cache`, which the
    nightly `audit-main` runs on `main`) failed that a cached pass, a suite's or a test's, answers for (a key that misses an input,
    or a test whose outcome isn't a function of its inputs); ``nondeterministic``, the tests that both passed and failed under one
    key (their inputs were identical)."""
    found = []
    for f in ("suites.json", "suites-lean.json"):
        if (out / f).is_file():
            found += [{"suite": d["suite"], field: d[field]} for d in json.loads((out / f).read_text()) if d.get(field)]
    return found


def write_result(run_dir: Path, ok: bool, done: list[dict], verdicts: dict | None = None, contradicted: list[dict] = (),
                 nondeterministic: list[dict] = ()) -> None:
    doc = {"schema": "research/result/v0.1", "kind": "check/v1", "run_id": os.environ["RESEARCH_RUN_ID"],
           "workload_fingerprint": {"steps": [s["step"] for s in done]},
           "validation": {"status": "passed" if ok else "failed", "checker": "tools/check/check.py",
                          "detail": "; ".join(f"{s['step']} {s['status']}" for s in done)},
           "measurements": [{"name": f"{s['step']}_wall", "value": s["seconds"], "unit": "s"} for s in done],
           "steps": done, "artifacts": [p for p in ("suites.json", "suites-lean.json", "suites/", "circuit-check.json", "lean-audit/",
                                                    "verdicts.tar.gz", STEPS_LOG,
                                                    *(s.get("log") for s in done), *agreement_verdicts(run_dir))
                                        if p and (run_dir / p).exists()]}
    if verdicts:
        doc["verdicts"] = verdicts
    if contradicted:
        doc["cache_contradictions"] = list(contradicted)
    if nondeterministic:
        doc["nondeterministic"] = list(nondeterministic)
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def main(argv: list[str]) -> int:
    if argv == ["--agreement-files"]:
        from research.store.local import LocalStore

        pin = json.loads(UPSTREAM_PIN.read_text())
        print(" ".join(f"--send {LocalStore().fetch(pin[k]['art'])}" for k in ("upstream", "inputs")))
        return 0
    if argv == ["--lean-deps-files"]:
        urls = lean_deps_url_file(Path(tempfile.mkdtemp(prefix="check-lean-deps-")))
        print(f"--send {urls}" if urls else "")
        return 0
    if argv[:1] == ["--lean-agreement"] and len(argv) == 4:
        return lean_agreement(*map(Path, argv[1:]))
    if argv[:1] == ["--lean-slot"] and len(argv) > 1:
        cmd = lean_slot(argv[1:])
        os.execv(cmd[0], cmd)
    if argv == ["--preflight"]:
        return preflight_only()
    if "--record" in argv:
        if os.environ.get("RESEARCH_RUN_ID"):
            print("check: --record inside a recorded run", file=sys.stderr)
            return 2
        return record([a for a in argv if a != "--record"])
    if os.environ.get("RESEARCH_RUN_ID") and os.environ.get("VERITY_SKIP_STORE") == "1":
        print("check: VERITY_SKIP_STORE=1 inside a recorded run: a recorded check runs the store-backed tests (the module docstring's "
              "\"The evidence store\")", file=sys.stderr)
        return 2
    cache = "--no-cache" not in argv
    keep_going = "--keep-going" in argv
    rest = [a for a in argv if a not in ("--no-cache", "--keep-going")]
    pack = None
    if "--verdicts-in" in rest and rest.index("--verdicts-in") + 1 < len(rest):
        i = rest.index("--verdicts-in")
        pack, rest = Path(rest[i + 1]), rest[:i] + rest[i + 2:]
    n = None
    if rest[:1] == ["--cores"] and len(rest) == 2 and rest[1].isdigit():
        n, rest = int(rest[1]), []
    if rest:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    run_dir = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else None
    out = run_dir or Path(tempfile.mkdtemp(prefix="check-"))
    verdicts = {"in": verdicts_in(pack, run_dir)} if pack and cache else {}
    before = VERDICTS.snapshot() if run_dir is not None and cache else None
    t0 = time.monotonic()
    ok, done = run(out, n, cache, keep_going)
    bad, flaky = contradictions(out), contradictions(out, "nondeterministic")
    for c in bad:
        print(f"check: CACHE CONTRADICTED in {c['suite']}: {', '.join(c['contradicts'][:5])} failed fresh, and a cached pass answers for it",
              flush=True)
    for c in flaky:
        print(f"check: NONDETERMINISTIC in {c['suite']}: {', '.join(c['nondeterministic'][:5])} passed and failed under one key", flush=True)
    if run_dir is not None:
        if before is not None:
            verdicts["out"] = {"file": "verdicts.tar.gz", "entries": VERDICTS.export(run_dir / "verdicts.tar.gz", before)}
        write_result(run_dir, ok, done, verdicts, bad, flaky)
    print(f"check: {'passed' if ok else 'FAILED'} in {time.monotonic() - t0:.1f} s ({', '.join(s['step'] + ' ' + s['status'] for s in done)});"
          f" outputs in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
