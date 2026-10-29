"""`check`: the repository's one check, what a merge into `main` requires (`research merge`).

    uv run --extra torch-cpu python tools/verity/check/check.py    run every step here, in the checkout, with CPU-only torch
    ... [--keep-going]                                      run every step to its end, even after one has failed
    ... PART...                                             only those parts (`tests`, `circuits`, `lean`: each job's `part` in
                                                            ci.toml) and what their jobs need, after the preflight; outside a
                                                            recorded run, since no gate takes part of a check
    ... --deferred STEP,...                                 only those ci.toml jobs and what they need, after the preflight: the
                                                            post-landing run drive launches on `main`'s tip (`train.sh
                                                            --deferred`), recorded as `DEFERRED_KIND`, which no gate takes
    ... --no-regenerate                                     lean-audit fails on a lock record difference rather than passing on
                                                            a regenerated record (drive's after-landing run)
    ... --no-cache [--circuit-cache]                        reuse no pass (the nightly `audit-main`); with --circuit-cache,
                                                            circuit-check alone still reuses and stores its per-target passes,
                                                            and only its entries of a pack are imported: drive's after-landing
                                                            run, where a cold `--all` (~4 h) outlasted every run
    uv run python tools/verity/check/check.py --record [--on M]    the same through `research run --tool check` (on machine M, a
                                                            multi-core CPU pod): an Attempt of the clean commit checked out, the
                                                            record `research merge` reads; with the pinned upstream build and
                                                            inputs when the commit changes verity/ml/flock/ against --base REF
                                                            (default origin/main), and presigned URLs for the pinned Lean
                                                            dependency bundles; with --on, in a free check slot of M
                                                            (`tools/verity/check/slot.py`), the lock and cores the train checks take,
                                                            and in the run's own clone of the shipped tree (`--cwd clone`);
                                                            --priority takes the next free slot ahead of waiting lane checks;
                                                            --agreement sends the upstream build whatever the commit changes
                                                            (a train that changes a guarantee elsewhere: `train.sh`);
                                                            a warning first when the commit can't land on REF as it is;
                                                            --on auto (AUTO: vy-nebius-2) or --on M1,M2 picks M
                                                            from `--slots` (`choose`): the first where a check slot no window
                                                            blocks and a Lean `check` slot are free, else the one with the
                                                            fewest waiters ahead (priority ones only, with --priority), and
                                                            refuses when no machine's slots can be read; with no slot free on
                                                            any, twins: the check on each machine it can run on (`twins`),
                                                            the first to take a slot runs; exit 75 when none ran: rerun
    uv run python tools/verity/check/check.py --record --on M --pool NAME [--deferred STEP,...]
                                                            the same in a slot of M's NAME pool (`research run --pool`'s pools,
                                                            e.g. `--on vy-nebius-cpu-1 --pool research`, any lane's), not its
                                                            merge-check slots, which on the CPU nodes take train checks alone:
                                                            a lane's own recorded check.  Launch it from a checkout, never inside
                                                            a `research run`, where --record is refused with the command to run
    uv run python tools/verity/check/check.py --slots [M ...]      each machine's check slots (holders, blocking and upcoming
                                                            windows), check line, and Lean audit slots and their waiters, read
                                                            over ssh by this tree's slot.py and lean_slot.py (`--status --json`);
                                                            then the machine --on auto would take; --json, the documents
    uv run python tools/verity/check/check.py --plan [COMMIT] [--json] [--on MACHINE]
                                                            each part's key at COMMIT (default HEAD) and whether a recorded
                                                            pass under it is here or in a pack `gather_verdicts` would take;
                                                            runs nothing (`plan.py`), read by `research merge --verdicts`;
                                                            --on: planned on that check node, a short `research run`
    uv run python tools/verity/check/check.py --lean-deps-files    the `--send` for those URLs, for `research merge --train`
    python tools/verity/check/check.py --preflight                 the preflight alone, for `research merge --train --prepare`
    ... [--verdicts-in PACK]                                import the passes another machine stored (`verdicts.py`) before
                                                            the steps; with --record, added to the pack it gathers from the
                                                            newest recorded checks, and with --on that pack travels with the run
    uv run python tools/verity/check/check.py --record --shard K/N --on M
                                                            shard K of ci.toml's plan with N shards, on machine M (each shard
                                                            on a pod of the lander's choosing): an Attempt no gate takes alone
    uv run python tools/verity/check/check.py --record --combine RUN...
                                                            the shards RUN..., one per K, recorded as one check of this commit
                                                            when they are together a passing one (`research merge` reads them)

Sharding.  `check.py --shard K/N` runs the preflight and then only shard K's parts of the plan with N shards that `ci.toml`
declares (`declaration.py`: each job in one shard, `lean-audit` split by Lake package), so one commit's check runs on N pods at
once and takes about as long as its longest shard. Its result is a `check-shard/v1` (`SHARD_KIND`) that names the shard, the
commit, the config (the tree of `tools/verity/check/` at that commit), the shard's parts and every part an unsharded check covers, and
each step the parts it ran. `--record --combine RUN...` records, on the clean commit, a `check/v1` whose steps are the shards'
parts, each once (the preflight's steps once, passed in every shard), when `research.merge.shard_set` accepts them: the
shards' Attempts are passing runs of the declared command with `--shard K/N` on that clean commit, their results passed with
no step failed or stopped, all of one plan (one N, K = 1..N once each) and one config (the commit's), and their parts are
every part, each once, covering every job the commit's `ci.toml` declares. `research merge` takes the combined Attempt only
after judging its shards again the same way and finding its steps theirs, so a missing, repeated, failed or mismatched shard
passes nothing. A step skipped in its shard (lean-agreement without the upstream build) is skipped in the combined check too.

First the preflight, seconds in all: whether `uv.lock` is current (`uv lock --check`), the repository's wall-clock lint and its
tree's own tests (`LINTS`: markdown caps, report-genre names, secret-like literals, blob and data budgets). The groups start only
once it passes, so a check that one of them fails ends in seconds, not hours.

A step that fails stops the others: each running step's processes get SIGINT (Python code runs its `finally`s, so the Lean audit
removes its scratch tree) and whatever is left of them SIGKILL after `STOP_GRACE_S`, and no group starts another step. Such a
step's status is `stopped`. `--keep-going` lets every group run to its end instead, keeping the verdicts its passing steps store,
except after a failed preflight, which starts no group, or a step in `HALTS`: a Lean module that doesn't compile, or a guarantee whose
record is stale, fails the audit too, so nothing still running is worth the slot; likewise a Rust crate that doesn't compile
(`HALTS_ON`: flock-circuit-build's log says `could not compile`) fails flock-rust-tests.
A step whose outputs (its log, the run dir's files named after it, `WATCH`'s) haven't changed for `IDLE_S`, or that is still
running after `STEP_BUDGET_S`, is stopped the same way (SIGTERM, then SIGKILL) and fails, `hung: no output for N min` or `over
budget`, so a hung test can't hold a check slot for a day; each suite is bounded first by its own `--timeouts`.
The steps stay in this process's group, which a runner's timeout and a manual stop signal as a whole. Each step's record is
appended to `steps.jsonl` in the output directory as the step ends, so a failure shows while the rest still run, and mirrored
as a trace/v1 span line in `spans.jsonl` (`spans.py`: the run opens a root span, each step is a completed child, the run's end
completes the root), the frozen shape the observability watcher reads.  The run's live record (`live.py`, `check-live.json`)
holds every planned step's state, elapsed time and progress, and a recorded run publishes it on the store's remote with its
custody key as each step ends and every few minutes, so `verity ci ls` and `verity ci inspect RUN` read a running check from
anywhere without its node.  A SIGTERM
(`research cancel`, a runner's timeout) stops the steps the same way (a step it reaches first and kills is `stopped` too, not
failed), and a recorded run still writes its result, validation `not_run`, and the pack of the passes its steps stored before
it, then exits 143: a cancelled check's passes are reused.

Then the independent groups run at once, the cores split between the first two (`--cores N`, default the CPUs this process may
use). Which group a step is in, its order there, a step in another group it waits on, and whether its failure halts the run
come from `tools/verity/check/ci.toml` (`declaration.py`); this file builds each step's command, and refuses one the declaration
doesn't name:

    pytest          every test suite (`tools/verity/check/suites.py`) but those that use Lean, half the cores' worth at once: each
                    package's tests in their own process, a pass reused while its inputs are unchanged; each suite's log and the
                    outcomes (`suites.json`) beside the run's result
    circuit-check   first flock-circuit-build, `verity/ml/flock/check_build.sh`: `cargo check` of flock-circuit in every CPU
                    feature set a pod builds, the proving builds without `seed-injection` among them (a Flock b684b12 checkout,
                    cached under $FLOCK_CHECKOUT), so a merge that breaks the Rust build ends the run in minutes; then
                    `circuit-check --all --jobs` the other half, whose workers start only while their measured peaks
                    (`circuit_check/heavy.json`) fit the memory the run has, at most its check slot's mem= (`--mem-gb`, from
                    slot.py's CHECK_SLOT_MEM_GB) less `SIBLINGS_GB` when pytest or Lean runs beside it, less pytest's share
                    alone when no Lean step does (`cc_mem_gb`), and all of it once every step beside it has ended
                    (`grant_when_alone`): every registered Definition and template
                    (`tools/verity/circuit_check`),
                    its report beside the run's result; then flock-rust-tests, `check_build.sh test` on that checkout:
                    `cargo test --release` of flock-live's unit tests and of the five upstream crates the patches change
    lean            `lake build` of the Lean verifier (`verity/core/service/flock`), then its partition check against
                    `verity.ml.circuits.partition` (`unit_cut_agree.py`), then its agreement with upstream's Rust verifier (`ci.py`) on
                    the upstream build stored once and pinned by hash (`verity/ml/flock/verifier/upstream.json`), when the run has
                    it; skipped by name without it.  `research merge` requires it for a change under verity/ml/flock/ (`tool.py`)
    lean-suites     once lean-build has passed, beside the rest of the lean group: the suites that use Lean (`AFTER_LEAN`,
                    outcomes in `suites-lean.json`) on the verifier it built
    lean-audit      at once with the lean group: the audit of every Lake package with a `lean-audit.json` (`lean_audit.py`,
                    which runs `tools/verity/lean/check.py` on scratch trees of its own: axioms, sorry, declared axioms, escapes,
                    files, kernel bypasses, code run while compiling, dependency .olean digests, the guarantees' records, a
                    kernel replay; its reports under `lean-audit/`).  It reads only the source tree and builds nothing in the
                    checkout, so the lean group's steps never wait on it, nor it on them
    lean-changed    at once with the lean group: `lake build` of each Lake package whose audit misses, on a tree this machine
                    keeps built (`lean_changed.py`), then the audit's judgment of its guarantees on that build without the
                    kernel replay, so a module that no longer compiles or a guarantee's record a merge left stale ends the
                    run in minutes, not when the audit's cold build reaches it; it skips what it can't build or read here and
                    vouches for nothing: the audit decides

Caches (``$CHECK_CACHE``, default ``~/.cache/verity-check``; ``--no-cache`` for a cold run), each a hit only on identical inputs:

    pytest          `suites.py`'s own: a suite's pass is reused, on any machine, while its declared inputs, the base image and the
                    check code are unchanged, and names the recorded run that earned it (`--no-cache` passes `--fresh`)
    circuit-check   `circuit-check --cache`, per target: a target's result is reused when every file, directory listing,
                    existence probe and loaded library its own check read or resolved (the Definitions it reached, their builds
                    and modules' import closures, its bindings and its pins, in `targets.py` or a Definition module's sidecar) is
                    unchanged, with the checker's own version; one after a read the tracer cannot see (a program it starts) is
                    not stored (`circuit_check.cache`); `--no-cache --circuit-cache` keeps this one cache on
    lean-audit      per package (`lean_audit.py`): a pass is reused while the content of the package, its path dependencies,
                    `tools/verity/lean/` and its toolchain and manifest is unchanged, wherever the package is (a move is no miss, and
                    its kept build follows it); a miss is audited on a scratch tree holding only those, so a read outside them
                    fails there; fetched dependencies stay warm on the machine under manifest + toolchain, and a machine without
                    them restores the bundle `lean-deps.json` pins from the store, not from GitHub; a step that audits any
                    package first waits for one of the node's Lean audit slots (`lean_slot.py`). The step fails on a fast-path
                    result in its `audit.json`, cached or not (`lean_fast`): no merge check accepts an audit without the kernel
                    replay
    lean-changed    not a verdict: its tree under `lean-changed/` keeps each package's last build there, so Lake rebuilds only
                    what changed and what imports it, and its fetched dependencies are hard links to lean-audit's, read-only
    lean-agreement  a pass is reused when everything the job reads is the same (`agreement_key`: verity/ml/flock/verifier/'s
                    tracked files, which pin the upstream build and inputs, the interpreter, the environment)
    flock-rust-tests  likewise (`rust_tests_key`: the script, which pins the Flock commit and the Rust toolchain, the patches,
                    the live crate, that toolchain's `rustc -vV`, the environment), so only a change to those pays for the Rust
                    tests
    preflight-lints likewise (`lints_key`: the commit's tree, which they read whole, the interpreter, the environment); a tree
                    with uncommitted changes has no key, and runs them

Inside a recorded run it writes `$RESEARCH_RUN_DIR/result.json` (kind `check/v1`): each step's status, seconds and cache use,
with each step's log beside it, and `verdicts.tar.gz`: the passes this run stored, which another machine imports with
`--verdicts-in`, written again as each step ends, every few minutes and on SIGTERM (`Packs`), so a cancelled check's pack holds
what its steps had passed.  `--record` declares it an output, so custody uploads it however the run ends: a cancel's SIGKILL
leaves no result.json, and the pack is then in the attempt's run record alone (`attempt_pack` reads either).  Custody misses it only
where it uploads nothing: `--no-custody-r2` and a run without `--on`, and a run whose runner is killed or whose machine is lost.  `--record` without `--no-cache` (or with `--circuit-cache` beside it, for circuit-check's entries alone) sends one pack of the newest recorded checks' packs, failed and cancelled
ones' too, since a pack holds only passes (`gather_verdicts`), with a
`--verdicts-in` pack added to them, never in their place, so every machine reuses every pass a recorded check earned; the pack
imported is named in the result's `verdicts`.

The evidence store.  A recorded check runs the store-backed tests (integrations/vllm's `store_io`) on the store's remote, never
skipped: its steps get VERITY_STORE_REQUIRED=1 (a missing remote fails those tests) and, as CHECK_STORE_CUSTODY, the run's custody
dir (`research run --custody-r2`, the default with --on: a delete-free key with GetObject), which `store_io` hands to its store
CLI only.  So `--record --on` must be launched where the parent R2 key is (the Cursor secrets, or `~/.config/verity/r2.env` when it
holds no AWS_SESSION_TOKEN): a temporary credential cannot mint the custody key, and `research run` refuses.  VERITY_SKIP_STORE=1
is refused inside a recorded run.  Outside one, the store-backed tests skip where no remote is configured.  Under
VERITY_STORE_REQUIRED=1 the suites' guard fails any skip whose reason says "the evidence store is not reachable", whichever helper
fetched (C-Flock's `verifier/ci.py` hands the custody key to its store CLI too).

The vLLM reference regression.  A recorded run's steps also get VERITY_REGRESSION=1, so integrations/vllm's suite runs
`tests/regression`'s T0 tier against the frozen rows in the store (about four minutes of tests): every check but the manifest
rebuilds, which take an hour and run after each train on main instead (`post_train.py`).  Under VERITY_STORE_REQUIRED=1 a row's
input or contract that doesn't resolve fails there; only a check `fixtures.toml` retires on a row, with its reason, skips.
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
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: `ci.py`, the agreement job of the Lean verifier against upstream's Rust verifier, and its other scripts
FLOCK_VERIFIER = ROOT / "verity" / "ml" / "flock" / "verifier"
#: the Lean verifier of record, the Lake package `flock-verify` is built from (a join of its own, so a layout move that takes
#: the Lean package away from the scripts rewrites it)
FLOCK_LEAN = ROOT / "verity" / "core" / "service" / "flock"
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
#: integrations/vllm's reference regression (`tests/regression`): on in a recorded run, its T0 tier less the manifest rebuilds
REGRESSION_ENV = "VERITY_REGRESSION"


def custody_dir() -> Path | None:
    """The recording run's custody dir (`<root>/requests/<run id>/custody/`, beside `<root>/runs/<run id>/`) when it holds a key."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return None
    d = Path(run_dir).resolve().parent.parent / "requests" / run_id / "custody"
    return d if (d / "store.toml").is_file() and (d / "cred.json").is_file() else None


def _load(name: str, d: Path | None = None):
    spec = importlib.util.spec_from_file_location(f"check_{name}", (d or Path(__file__).resolve().parent) / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m                       # dataclasses resolve their module by name
    spec.loader.exec_module(m)
    return m


DECLARATION = _load("declaration")
#: the jobs `tools/verity/check/ci.toml` declares: their groups, order within a group, and what halts the run (`groups`, `HALTS`)
JOBS = DECLARATION.load()


#: the verdict packs (`verdicts.py`)
VERDICTS = _load("verdicts")
#: the helpers `lean_audit.py` shares (`common.py`), kept out of this file so that its audit key doesn't cover this file
COMMON = _load("common")
#: trace/v1 spans over the run (`spans.py`): `spans.jsonl` beside `steps.jsonl`, the frozen shape the watcher reads
SPANS = _load("spans")
#: the run's live record (`live.py`): `check-live.json` beside `steps.jsonl`, published on the store's remote as the run goes
LIVE = _load("live")
#: the test runner (`suites.py`), for where the steps' Python imports this tree's packages from
SUITES = _load("suites")
cache_dir = COMMON.cache_dir
cores = COMMON.cores
_sha256 = COMMON.sha256
tracked_files = COMMON.tracked_files


ELAN = SUITES.ELAN


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
#: the parts a recorded shard runs (none for a combine), among its inputs: the preflight asks only for the Lean dependencies of
#: the packages it audits, since it carries URLs for no others
SHARD_PARTS = "check-shard-parts.json"
#: what a `--record --on` check runs under on the machine: a free check slot's lock and cores (slot.py), from the shipped tree
SLOT_WRAP = ["python3", "tools/verity/check/slot.py", "--"]


def slot_wrap(priority: bool = False, twin: str | None = None) -> list[str]:
    """SLOT_WRAP naming who asked (RESEARCH_LANE, else RESEARCH_AGENT_URL), which `slot.py --status` on the machine shows, and
    with --priority for the lander's train checks, --rank from CHECK_SLOT_RANK (drive's train position, so the train that lands
    next takes the next slot) and --twin for one of a check's twins."""
    by = os.environ.get("RESEARCH_LANE") or os.environ.get("RESEARCH_AGENT_URL")
    rank = os.environ.get("CHECK_SLOT_RANK", "")
    return [*SLOT_WRAP[:-1], *(["--priority"] if priority else []),
            *(["--rank", rank] if priority and re.fullmatch(r"\d{1,6}", rank) else []), *(["--by", by] if by else []),
            *(["--twin", twin] if twin else []), SLOT_WRAP[-1]]


def twins(names: list[str], docs: dict[str, dict | str], home: str, priority: bool = False) -> list[str]:
    """The machines among ``names`` to launch twins of a check on (`slot.py --twin`), ``home`` first: every one a check can run
    on (`placement` refuses none) when ``home``, `choose`'s pick, has no slot free for it; else none, and it runs on ``home``."""
    def runs(name: str) -> bool:
        try:
            placement(docs[name], priority)
        except (ValueError, KeyError, TypeError, AttributeError):
            return False
        return True

    if placement(docs[home], priority)[0]:
        return []
    out = [home, *(n for n in names if n != home and runs(n))]
    return out if len(out) > 1 else []


#: the machines `--on auto` and `--slots` consider, in this order: the shared nodes' merge-check slots (the CPU nodes' are
#: only=trains)
AUTO = ("vy-nebius-2",)
#: where a lane's own recorded check goes when it names no machine (`nested_record`): a pool any lane takes, one of
#: research.auto's SPILL
LANE_POOL = ("vy-nebius-cpu-1", "research")
#: how long reading one machine's slots may take, ssh included
SLOTS_TIMEOUT_S = 60
#: a full check's length: `--on auto` places none on a machine whose deadline (/etc/research/deadline) is nearer than this
FULL_CHECK_S = 9000
#: a lane's own test run, which no landing needs (a PR lands on an eyeball review)
LANE_RUN = ('`research run --on vy-nebius-cpu-K --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && '
            'uv run tools/verity/check/suites.py --changed $(git rev-parse origin/main) --quick"`')
#: what `--slots` runs on a machine as `python3 -`, after SLOT_SRC and LEAN_SRC: both `--status --json` documents as one object,
#: each `{"error": ...}` when it can't be read there, and the seconds left until the machine's DEADLINE (null without one)
SLOTS_PROGRAM = '''
import json, sys, time, types
from datetime import datetime, timezone


def load(name, src, file):
    m = types.ModuleType(name)
    m.__file__ = file
    sys.modules[name] = m
    exec(compile(src, file, "exec"), m.__dict__)
    return m


def doc(f):
    try:
        return f()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


now = datetime.now(timezone.utc)
slot = load("check_slot_for_lean_slot", SLOT_SRC, "tools/verity/check/slot.py")
lean = load("check_lean_slot", LEAN_SRC, "tools/verity/check/lean_slot.py")
try:
    left = int(open("/etc/research/deadline").read().split()[0]) - time.time()
except (OSError, ValueError, IndexError):
    left = None
print(json.dumps({"check": doc(lambda: slot.status_doc(now)), "lean": doc(lambda: lean.status_doc(now)), "deadline_in_s": left}))
'''
SLOT = _load("slot")


def slots_payload() -> bytes:
    """`SLOTS_PROGRAM` with this tree's slot.py and lean_slot.py (both stdlib only) inlined, so neither need be on the machine;
    lean_slot.py takes the slot.py loaded before it (`SLOT_MODULE`)."""
    here = Path(__file__).resolve().parent
    return (f"SLOT_SRC = {(here / 'slot.py').read_text()!r}\nLEAN_SRC = {(here / 'lean_slot.py').read_text()!r}\n"
            + SLOTS_PROGRAM).encode()


def reach(name: str, machines: dict | None = None) -> list[str]:
    """The argv that runs `python3 -` on machine ``name`` as `research run --on` reaches it: its registry entry, bound to project
    verity, and its ssh target (`research.remote`)."""
    from research import remote

    t = remote.pod_target(remote.resolve_machine(name, "verity", machines))
    return [*remote.ssh_argv(t["host"], t["port"], user=t["user"]), "python3 -"]


def _ssh(argv: list[str], payload: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=payload, capture_output=True, timeout=SLOTS_TIMEOUT_S)


def read_slots(names: list[str], reach=reach, run=_ssh) -> dict[str, dict | str]:
    """Each machine's `--status --json` documents (`SLOTS_PROGRAM`), read at once, or why they couldn't be: not a machine this
    project may use, ssh failing or silent for SLOTS_TIMEOUT_S, an answer that isn't JSON."""
    payload = slots_payload()

    def one(name: str) -> dict | str:
        try:
            r = run(reach(name), payload)
        except subprocess.TimeoutExpired:
            return f"no answer in {SLOTS_TIMEOUT_S} s"
        except Exception as e:                                  # not listed, another project's, its pod gone: never chosen
            return str(e) or type(e).__name__
        if r.returncode:
            err = r.stderr.decode(errors="replace").strip().splitlines()
            return f"exit {r.returncode}: {err[-1] if err else 'no output'}"
        try:
            doc = json.loads(r.stdout)
        except ValueError:
            return f"its answer is not JSON: {r.stdout[:120]!r}"
        return doc if isinstance(doc, dict) else f"its answer is not a JSON object: {r.stdout[:120]!r}"

    with ThreadPoolExecutor(max_workers=max(1, len(names))) as ex:
        return dict(zip(names, ex.map(one, names)))


def placement(doc: dict, priority: bool = False) -> tuple[bool, int, str]:
    """A check's prospects on a machine whose `read_slots` documents are ``doc``: whether a check slot and a Lean `check` slot
    are free for it (more free than live waiters ahead of it), how many live waiters are ahead of it in the check line (with
    ``priority``, a train check's, only the priority ones), and the state in a few words. ValueError when no check can run
    there: a document has an error, the slots file lists no slot, the Lean slots file grants check's lean-audit step none, or the
    machine's deadline is nearer than FULL_CHECK_S."""
    check, lean = doc["check"], doc["lean"]
    for d, what in ((check, "check slots"), (lean, "Lean audit slots")):
        if "error" in d:
            raise ValueError(f"its {what} can't be read: {d['error']}")
    if (left := doc.get("deadline_in_s")) is not None and left < FULL_CHECK_S:
        raise ValueError(f"its deadline is {max(left, 0) / 60:.0f} min away, under the {FULL_CHECK_S / 60:.0f} min a full check takes")
    ahead = len([w for w in check["waiters"] if w["priority"] or not priority])
    waiting = f"{ahead} waiting ahead" + (" with priority" if priority else "")
    if check["slots"] is None:
        room, why = True, f"no check slots ({check['file']}): a check runs unpinned at once"
    elif not check["slots"]:
        raise ValueError(f"{check['file']} lists no slots")
    else:
        mine = [s for s in check["slots"] if priority or not s.get("trains")]
        if not mine:
            raise ValueError(f"every slot in {check['file']} is for train checks (only=trains). A PR lands on an eyeball review, "
                             "and no lane check is required; to run its tests anyway: " + LANE_RUN)
        free = [s for s in mine if s["holder"] is None and not s["blocked"]]
        room = len(free) > ahead
        why = "check slots: " + ", ".join(f"{s['name']} " + ("held" if s["holder"] else "blocked" if s["blocked"] else "free")
                                          for s in mine) + f"; {waiting}"
    if lean["pools"] is None:
        return room, ahead, f"{why}; no Lean audit cap"
    pool = next((p for p in ("check", "any") if p in lean["pools"]), None)
    if pool is None:
        raise ValueError(f"{lean['file']} lists neither pool check nor any, so check's Lean audit would be refused")
    p = lean["pools"][pool]
    free_lean = len([s for s in p["slots"] if s["holder"] is None and not s["past_n"]])
    why += f"; Lean {pool}: {free_lean} of {p['n']} free" + (f", {len(p['waiters'])} waiting" if p["waiters"] else "")
    return room and free_lean > len(p["waiters"]), ahead, why


def choose(names: list[str], docs: dict[str, dict | str], priority: bool = False) -> tuple[str | None, str, list[str]]:
    """`--on auto`'s machine among ``names``, its reason, and each machine skipped with why. The first machine where a check slot
    with no window blocking it and a Lean `check` slot are free for this request (`placement`); else the one with the fewest
    live waiters ahead of it, the first listed among those tied. A machine whose documents can't be read is skipped; with none
    left, no machine (the caller refuses: a guess could put a check into a timed window)."""
    ranked, skipped = [], []
    for i, name in enumerate(names):
        doc = docs.get(name)
        try:
            if not isinstance(doc, dict):
                raise ValueError(doc or "not read")
            room, ahead, why = placement(doc, priority)
        except ValueError as e:
            skipped.append(f"{name}: {e}")
            continue
        except (KeyError, TypeError, AttributeError) as e:
            skipped.append(f"{name}: its slots documents are malformed ({type(e).__name__}: {e})")
            continue
        ranked.append(((0, 0, i) if room else (1, ahead, i), name, room, why))
    if not ranked:
        return None, "no machine's slots could be read", skipped
    _, name, room, why = min(ranked)
    lead = ("a check slot and a Lean check slot are free" if room else
            "the fewest waiting ahead, as no machine has a check slot and a Lean check slot free")
    return name, f"{lead} ({why})", skipped


#: the checks `--on auto` or `--on M1,M2` placed from this launcher and hasn't yet seen in their machine's check line, by commit:
#: until one shows there (or PLACED_S passes: train.sh's wait for a launch), it counts as a priority waiter on its machine, so
#: checks placed at once go to different machines (23 Oct 13:04Z: drive's four parallel launches all read cpu-1's one free
#: slot as theirs, and three queued there for an hour while cpu-4's stood free)
PLACED = Path(tempfile.gettempdir()) / "verity-check-placed.json"
PLACED_S = 1800


def _trees(doc: dict) -> set[str]:
    """The shortened commits a machine's check slots hold or wait for: each one's commit, else its tree's name (<root>/src/<sha>,
    from a slot.py that records no commit)."""
    check = doc.get("check") or {}
    who = [s["holder"] for s in check.get("slots") or [] if s.get("holder")] + list(check.get("waiters") or [])
    return {(h.get("commit") or Path(h["tree"]).name)[:12] for h in who if h.get("commit") or h.get("tree")}


def place(names: list[str], commit: str, priority: bool = False, read=None, path: Path | None = None,
          now=time.time) -> tuple[dict[str, dict | str], str | None, str, list[str]]:
    """`read_slots` and `choose` for ``commit``, one launcher process at a time, with the other checks this launcher placed and
    no machine shows yet (PLACED) as waiters on theirs; ``commit`` is recorded on its pick."""
    import fcntl

    path = path or PLACED
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        docs = (read or read_slots)(names)
        try:
            placed = json.loads(path.read_text())
        except (OSError, ValueError):
            placed = {}
        t = now()
        placed = {c: p for c, p in placed.items() if c != commit and t - p["at"] < PLACED_S
                  and not (isinstance(d := docs.get(p["on"]), dict) and c[:12] in _trees(d))}
        for c, p in placed.items():
            if isinstance(d := docs.get(p["on"]), dict) and isinstance((d.get("check") or {}).get("waiters"), list):
                d["check"]["waiters"].append({"priority": True, "pid": 0, "since": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(p["at"])),
                                              "minutes": int((t - p["at"]) // 60), "tree": c, "by": "a launch placed here"})
        got, why, skipped = choose(names, docs, priority)
        if got:
            placed[commit] = {"on": got, "at": t}
            path.write_text(json.dumps(placed))
    return docs, got, why, skipped


def _who(h: dict) -> str:
    """A holder or waiter in a few words: since when, its run, its tree (a commit's, shortened) and who asked (a URL's last part)."""
    parts = [f"since {h['since']} ({h['minutes']} min)" if h.get("since") else f"pid {h['pid']}"]
    parts += [f"run {h['run']}"] if h.get("run") else []
    tree = h.get("commit") or (Path(h["tree"]).name if h.get("tree") else "")
    parts += [f"tree {tree[:12] if re.fullmatch(r'[0-9a-f]{40}', tree) else tree}"] if tree and tree != h.get("run") else []
    parts += [f"by {h['by'].rstrip('/').rsplit('/', 1)[-1]}"] if h.get("by") else []
    return ", ".join(parts if len(parts) > 1 or not h.get("cmd") else [*parts, h["cmd"][:60]])


def slots_block(name: str, doc: dict | str) -> list[str]:
    """One machine's lines in `--slots`: each check slot (free, or its holder) with what blocks it and its windows ahead, the check
    line's waiters, and each Lean pool's free slots, holders and waiters."""
    if not isinstance(doc, dict):
        return [f"{name}: can't be read: {doc}"]
    check, lean = doc.get("check") or {}, doc.get("lean") or {}
    out = [f"{name} (at {check.get('now') or lean.get('now') or '?'})"]
    if "error" in check:
        out.append(f"  check slots: can't be read: {check['error']}")
    elif check.get("slots") is None:
        out.append(f"  no check slots ({check.get('file')}): a check runs unpinned at once")
    else:
        for s in check["slots"]:
            out.append(f"  check {s['name']}  cores {s['cpus']}  " + (f"held {_who(s['holder'])}" if s["holder"] else "free")
                       + (f"; blocked: {s['blocked']}" if s["blocked"] else ""))
            out += [f"    {SLOT.window_text(w)}" for w in s["upcoming"]]
        ws = check["waiters"]
        pri = len([w for w in ws if w["priority"]])
        out.append("  check line: " + (f"{len(ws)} waiting" + (f" ({pri} with priority)" if pri else "") + ": "
                                       + "; ".join(_who(w) + (", priority" if w["priority"] else "") for w in ws) if ws else "nobody waiting"))
    if "error" in lean:
        out.append(f"  Lean audit slots: can't be read: {lean['error']}")
    elif lean.get("pools") is None:
        out.append(f"  Lean audit slots: no cap ({lean.get('file')}): audits run unslotted")
    else:
        for pool, p in lean["pools"].items():
            free = len([s for s in p["slots"] if s["holder"] is None and not s["past_n"]])
            held = [f"{pool}/{s['index']} {_who(s['holder'])}" for s in p["slots"] if s["holder"]]
            out.append(f"  Lean {pool}: {free} of {p['n']} free" + "".join(f"; {h}" for h in held)
                       + (f"; {len(p['waiters'])} waiting: " + "; ".join(map(_who, p["waiters"])) if p["waiters"] else ""))
    return out


def slots_view(argv: list[str], read=None) -> int:
    """`--slots [--json] [--priority] [MACHINE ...]`: `slots_block` for each machine (default AUTO), then the machine
    `--on auto` (or `--on M1,M2`) would take now; --json, the documents themselves. 1 when no machine could be read."""
    names = [a for a in argv if a not in ("--json", "--priority")] or list(AUTO)
    if any(a.startswith("-") for a in names):
        print("usage: check.py --slots [--json] [--priority] [MACHINE ...]", file=sys.stderr)
        return 2
    docs = (read or read_slots)(names)
    if "--json" in argv:
        print(json.dumps(docs, indent=1))
    else:
        for name in names:
            print("\n".join(slots_block(name, docs[name])))
        got, why, _ = choose(names, docs, "--priority" in argv)
        on = "auto" if names == list(AUTO) else ",".join(names)
        print(f"--on {on}{' --priority' if '--priority' in argv else ''} would " + (f"take {got}: {why}" if got else f"refuse: {why}"))
    return 0 if any(isinstance(d, dict) for d in docs.values()) else 1


def take_lean_deps_urls(out: Path) -> Path | None:
    """The presigned URLs sent with the run, moved out of the run directory, where the steps write, before any step starts: each
    grants a read until it expires. Custody deletes one left there before it takes the run's record
    (`research.store.custody.CREDENTIALS`)."""
    sent = out / "inputs" / LEAN_DEPS_URLS
    if not sent.is_file():
        return Path(os.environ[LEAN_DEPS_URLS_ENV]) if os.environ.get(LEAN_DEPS_URLS_ENV) else None
    fd, tmp = tempfile.mkstemp(prefix="lean-deps-urls-", suffix=".json")
    with os.fdopen(fd, "wb") as f:
        f.write(sent.read_bytes())
    sent.unlink()
    return Path(tmp)


#: Lean's job code, which the lean-audit and lean-changed steps run (`lean_audit.py`, `lean_changed.py`)
LEAN_JOBS = ROOT / "tools" / "verity" / "lean"


def _lean_audit():
    spec = importlib.util.spec_from_file_location("check_lean_audit_urls", LEAN_JOBS / "lean_audit.py")
    la = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(la)
    return la


def lean_deps_url_file(dest: Path) -> Path | None:
    """`lean-deps-urls.json` for this tree's pinned Lean dependency bundles, signed by this machine's store remote."""
    return _lean_audit().url_file(dest)


#: the lean-audit step's wrapper when it audits anything: one of the node's Lean audit slots, from pool `check` (lean_slot.py)
LEAN_SLOT_WRAP = [sys.executable, str(ROOT / "tools" / "verity" / "check" / "lean_slot.py"), "--pool", "check", "--"]


def lean_regenerates() -> bool:
    """Whether this tree's lean-audit job takes `--regenerate`: a package whose only failures are record differences passes,
    and its report names the regenerated lock, which drive commits onto the train it lands.  Read from the job, so a
    tree from before it runs as it always did."""
    try:
        return '"--regenerate"' in (LEAN_JOBS / "lean_audit.py").read_text()
    except OSError:
        return False


def lean_audit_misses(cache: bool = True, only: list[str] | None = None) -> bool:
    """Whether the lean-audit step will audit a package (of ``only``, its `--package`s, when given): one whose pass isn't kept
    here under its key (`lean_audit.run`'s test)."""
    la = _lean_audit()
    files = tracked_files(ROOT)
    if not cache or files is None:
        return True
    pkgs = [p for p in la.packages(files) if not only or p in only]
    return any(not (cache_dir() / "lean-audit" / f"{la.key(p, files, la.FLAGS)}.json").is_file() for p in pkgs)


def lean_fast(out: Path) -> str | None:
    """Why the lean-audit step's pass is refused, or None: its `audit.json` can't be read, or holds a fast-path result
    (`lean_changed.py --records`) for a package, a report of kind `lean_changed.FAST_KIND` or a pass without the kernel replay."""
    p = out / "lean-audit" / "audit.json"
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return f"refused: no readable {p} ({type(e).__name__}) shows that every package had the full audit"
    reports = doc.get("packages", []) if isinstance(doc, dict) else None
    if not isinstance(reports, list) or not all(isinstance(r, dict) for r in reports):
        return f"refused: {p} is not an audit's report (an object whose packages are a list of objects)"
    kind = _load("lean_changed", LEAN_JOBS).FAST_KIND
    replayed = [isinstance(r.get("seconds"), dict) and "replay" in r["seconds"] for r in reports]
    fast = [str(r.get("package")) for r, rep in zip(reports, replayed) if r.get("kind") == kind or (not r.get("failures") and not rep)]
    if fast or doc.get("kind") == kind:
        return (f"refused: {', '.join(fast) or p}: a {kind} result (the fast path: no kernel replay, no tests), which no merge "
                "check accepts")
    return None


def lean_slot(cmd: list[str]) -> list[str]:
    """`--lean-slot CMD...`, the lean-audit step: CMD in one of the node's Lean audit slots when it will audit a package, as it
    is when it only reuses passes, which would otherwise wait behind an hour-long audit for seconds of work. CMD runs with
    LEAN_SLOT_POOL=check, so an audit it runs after all (a pass gone since) takes its slot from the same pool
    (`tools/verity/lean/check.py --build`)."""
    only = [cmd[i + 1] for i, x in enumerate(cmd[:-1]) if x == "--package"]
    return [*LEAN_SLOT_WRAP, *cmd] if lean_audit_misses("--fresh" not in cmd, only) else cmd


def requires_outside(lean: Path) -> list[str]:
    """The packages the Lean package ``lean`` requires from outside its directory, by its manifest or its lakefile: the
    agreement's pass is kept under that directory's files only, so it would outlive a change to one of them."""
    import tomllib

    m, lf = lean / "lake-manifest.json", lean / "lakefile.toml"
    entries = [*(json.loads(m.read_text()).get("packages", []) if m.exists() else []),
               *(tomllib.loads(lf.read_text()).get("require", []) if lf.exists() else [])]
    return sorted({e.get("name", "?") for e in entries
                   if (p := e.get("dir") or e.get("path")) is None or not (lean / p).resolve().is_relative_to(lean.resolve())})


def lean_steps(out: Path, cache: bool = True, urls: Path | None = None,
               regenerate: bool = True) -> list[tuple[str, list[str] | str, Path]]:
    """The Lean steps: nothing on a tree without the verifier (a skip, named); then `lake build`, the Lean partition check's
    conformance with `verity.ml.circuits.partition` (`unit_cut_agree.py`), the audit of every Lean package, and the agreement job
    against upstream's Rust verifier on the pinned upstream build (`upstream.json`) when the run has it; without it that step
    is skipped by name.  The audit regenerates lock records (`lean_regenerates`) unless ``regenerate`` is off (`--no-regenerate`,
    drive's after-landing run, where a record difference fails), and inside a recorded run it bundles the module versions it
    built for other nodes (`lean_audit.py --bundle`).  `research merge` requires it for a change under `verity/ml/flock/` (`tools/verity/check/tool.py`), and
    `--record` sends it then.  The agreement fails, sent or not, while the verifier's package requires one from outside its
    directory (`requires_outside`)."""
    lean = FLOCK_LEAN
    if not (lean / "lakefile.toml").exists():
        if FLOCK_VERIFIER.is_dir():
            return [("lean", f"the verifier is on this tree but its Lean package is not at {lean}", ROOT)]
        return [("lean", "skipped: verity/core/service/flock is not on this tree", ROOT)]
    verify = lean / ".lake" / "build" / "bin" / "flock-verify"
    agreement: list[str] | str = ("skipped: the pinned upstream build was not sent (a merge changing verity/ml/flock/ or "
                                  "verity/core/service/flock/ requires this step: `check.py --record` sends it)")
    if UPSTREAM_PIN.exists():
        got = agreement_files(out)
        if len(got) == 2:
            agreement = [sys.executable, str(Path(__file__).resolve()), "--lean-agreement", str(out), str(got["upstream"]), str(got["inputs"])]
        elif got:
            agreement = f"only {', '.join(got)} of the pinned upstream build and inputs was sent"
    if outside := requires_outside(lean):
        agreement = (f"the verifier's Lean package requires {', '.join(outside)} from outside its directory, which the "
                     f"agreement's key (`agreement_closure`) and `research merge`'s lean-agreement rule don't cover")
    return [("lean-build", ["bash", "-c", SUITES.LAKE_BUILD], lean),
            ("lean-unit-cut", [sys.executable, str(FLOCK_VERIFIER / "unit_cut_agree.py"), "--lean", str(verify),
                               "--out", str(out / "lean-unit-cut")], ROOT),
            ("lean-audit", [sys.executable, str(Path(__file__).resolve()), "--lean-slot",
                            sys.executable, str(LEAN_JOBS / "lean_audit.py"), "--out", str(out / "lean-audit"),
                            "--cache", str(cache_dir() / "lean-audit"), "--deps", str(cache_dir() / "lean-deps"),
                            *([] if cache else ["--fresh"]), *(["--urls", str(urls)] if urls else []),
                            *(["--bundle"] if os.environ.get("RESEARCH_RUN_ID") else []),
                            *(["--regenerate"] if regenerate and lean_regenerates() else [])], ROOT),
            ("lean-agreement", agreement, ROOT)]


def memory_gb(root: Path = Path("/")) -> int:
    """The memory this process may use: the least of MemAvailable and what each cgroup limit over it leaves, v2 or v1 (a pod's
/proc/meminfo is the host's; on a VM the limits are on its own cgroup's ancestors, such as a run's session scope). A v2
cgroup's page cache (active_file and inactive_file in its memory.stat) is in its memory.current, but the kernel reclaims it at
the limit, so it counts as left, as in slot.py's `memory_gb`: 27 Oct 00:48Z, vy-nebius-cpu-1's user slice held 336 GB of
cache in its 691 of 700 GB, and main's after-landing lean-agreement refused to run with "0 GB"."""
    def read(p: Path) -> str | None:
        try:
            return p.read_text().strip()
        except OSError:
            return None

    left: list[int] = []
    for line in (read(root / "proc/self/cgroup") or "").splitlines():
        if line.startswith("0::"):                     # cgroup v2: this process's cgroup, then each ancestor
            top = root / "sys/fs/cgroup"
            d = top / line[3:].strip("/")
            for c in (d, *[p for p in d.parents if p.is_relative_to(top)]):
                m, used = read(c / "memory.max"), read(c / "memory.current")
                if m and m.isdigit():
                    stat = dict(ln.split()[:2] for ln in (read(c / "memory.stat") or "").splitlines() if len(ln.split()) >= 2)
                    cache = sum(int(stat[k]) for k in ("active_file", "inactive_file") if stat.get(k, "").isdigit())
                    left.append(int(m) - int(used if used and used.isdigit() else 0) + cache)
    v1 = root / "sys/fs/cgroup/memory"
    m, used = read(v1 / "memory.limit_in_bytes"), read(v1 / "memory.usage_in_bytes")
    if m and m.isdigit() and int(m) < 1 << 60:         # v1 writes "no limit" as a number near 2^63
        left.append(int(m) - int(used if used and used.isdigit() else 0))
    for line in (read(root / "proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            left.append(int(line.split()[1]) << 10)
    return max(0, min(left)) >> 30 if left else 0


#: the share of this process's memory the agreement's verifier processes may count on: lean-suites (and, unsharded, the
#: suites and circuit-check) run beside it, and on a 128 GB pod the whole budget for the agreement OOM-killed 36 of them
#: (tip 38's shard 2/3, 6 Oct)
AGREEMENT_MEMORY_SHARE = 2


def session_jobs() -> int:
    """`ci.py --session-jobs`: its own default counts the host's CPUs and memory inside a container, so check passes what this
    process may use (its `AGREEMENT_MEMORY_SHARE`), at ci.py's `GB_PER_PROCESS` per verifier process."""
    m = re.search(r"^GB_PER_PROCESS = (\d+)", (FLOCK_VERIFIER / "ci.py").read_text(), re.M)
    return max(1, min(cores(), memory_gb() // AGREEMENT_MEMORY_SHARE // int(m.group(1) if m else 8)))


#: the agreement's memory floor: set 13's upstream replay (GEMM, m = 26) alone takes more than 15 GB
AGREEMENT_MIN_GB = 24


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def upstream_smoke(w: Path) -> list[tuple[str, list[str], dict]]:
    """One real replay per upstream binary that `vectors.json` gives a stored honest session (a set without lookup tables):
    (its pinned commit, the command `agree.py` would run, the verdict expected), over the unpacked build and inputs in ``w``."""
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
                f"machine, or rebuild on this glibc (verity/ml/flock/verifier/upstream-build.sh) and re-pin")
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
            print(f"check: lean-agreement: error: {src} is not the pinned {name} ({pin[name]['art']})", file=sys.stderr)
            return 1
        with tarfile.open(src) as t:
            t.extractall(w / name, filter="data")
    ups = []
    for c, h in sorted(pin["upstream"]["binaries"].items()):
        b = w / "upstream" / f"upstream-{c}"
        if not b.is_file() or _sha256(b) != h:
            print(f"check: lean-agreement: error: upstream-{c} does not match the pin", file=sys.stderr)
            return 1
        b.chmod(0o755)
        ups += ["--upstream", f"{c}={b}"]
    print(f"check: lean-agreement: unpacked the pinned upstream build ({len(ups) // 2} binaries) and inputs; preflight: one "
          f"stored honest session replayed per binary", flush=True)
    if why := agreement_preflight(pin, [w / "upstream" / f"upstream-{c}" for c in sorted(pin["upstream"]["binaries"])], w):
        print(f"check: lean-agreement: error: cannot run here: {why}", file=sys.stderr)
        return 1
    ci = pin.get("ci", {})
    sets = json.loads((FLOCK_VERIFIER / "vectors.json").read_text())["replayable"]["sets"]    # ci.py --jobs: sets at once
    argv = [sys.executable, str(FLOCK_VERIFIER / "ci.py"), "--lean", str(FLOCK_LEAN / ".lake" / "build" / "bin" / "flock-verify"),
            *ups, "--work", str(w / "work"), "--inputs", str(w / "inputs"), "--jobs", str(max(1, min(cores(), len(sets)))),
            "--session-jobs", str(session_jobs()),
            "--seed", str(ci.get("seed", 20260926)), "--fuzz", str(ci.get("fuzz", 20)), "--out", str(w)]
    print(f"check: lean-agreement: preflight passed; ci.py on {len(sets)} sets (it prints its results when every set ends)",
          flush=True)
    if rc := subprocess.run(argv, cwd=ROOT).returncode:
        print(f"check: lean-agreement: error: ci.py exited {rc}; its verdicts are in {w}", file=sys.stderr, flush=True)
    return rc


def landable(base: str) -> bool:
    """Whether `research merge` could land HEAD on ``base`` as it is: ``base`` is in HEAD, or merging HEAD into it leaves HEAD's own
    tree (the gate's `tree_identical`). Otherwise the tree that lands is a merge nobody checked yet."""
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT, capture_output=True).returncode == 0:
        return True
    r = subprocess.run(["git", "merge-tree", "--write-tree", base, "HEAD"], cwd=ROOT, capture_output=True, text=True)
    head = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return r.returncode == 0 and r.stdout.split("\n", 1)[0].strip() == head


def pool_env(pool: str | None, on: str | None, priority: bool) -> list[str]:
    """The `research run` options that put a `--record --on MACHINE --pool NAME` check in a slot of MACHINE's NAME pool, the
    pool `research run --pool NAME` takes (`slot_only.POOLS`), rather than its merge-check slots, which on the CPU nodes are
    only=trains: slot.py reads the pool's slots file from CHECK_SLOTS.  The command stays `slot_wrap`'s around the declared one,
    not `slot_only.in_pool`'s (`env ... nice -n 19`), so `research merge` and `gather_verdicts` read the attempt as a check of its
    commit.  ValueError when the pool can't be taken so."""
    if pool is None:
        return []
    from research.pods.nebius import slot_only

    if pool not in slot_only.POOLS:
        raise ValueError(f"{pool}: no such pool; the pools are {', '.join(sorted(slot_only.POOLS))}")
    if not on or on == "auto" or "," in on:
        raise ValueError(f"{pool} needs --on MACHINE, the one machine whose pool it waits in (`--on {LANE_POOL[0]} --pool "
                         f"{LANE_POOL[1]}`); --on auto and lists choose among merge-check slots")
    if priority:
        raise ValueError(f"{pool} takes no --priority: that is the lander's train checks' place in the merge-check slots' line")
    return ["--env", f"CHECK_SLOTS={slot_only.POOLS[pool]}"]


def nested_record(argv: list[str]) -> str:
    """The command that records ``argv``'s check from outside a `research run`: `record` launches a `research run` of its own,
    so inside one it is refused with this.  With no --on, on `LANE_POOL`, any lane's pool."""
    where = [] if "--on" in argv else ["--on", LANE_POOL[0], *([] if "--pool" in argv else ["--pool", LANE_POOL[1]])]
    return shlex.join(["uv", "run", "python", "tools/verity/check/check.py", "--record", *where,
                       *(a for a in argv if a != "--record")])


def record(argv: list[str]) -> int:
    """`--record [--on MACHINE|auto|M1,M2] [--base REF] [--agreement]`: `research run --tool check` on this clean commit, with the
    pinned upstream build and inputs (fetched from the store) when the commit changes a path `tools/verity/check/tool.py` names against
    REF (default origin/main, fetched first), or whenever --agreement is given, and the newest recorded checks' verdicts
    (`gather_verdicts`) unless it is given a pack or --no-cache without --circuit-cache. A commit that can't land on REF as it is
    (`landable`) gets a warning and is checked anyway. `--on auto` or a list takes the machine `choose` picks, first of all, or
    launches `twins` there and on the others (one `research run` each, waited on together: 0 when one passed, the exit of the
    one that ran otherwise, and slot.py's NOT_STARTED (75) when none ran, say with HOME unreachable: rerun it).  With
    `--deferred STEP,...` it is a `research run` without `--tool check` of those steps alone (no gate reads it), with the
    gathered verdicts and no upstream build or Lean dependencies.  `--pool NAME` (with `--on MACHINE`) waits for a slot of that
    machine's NAME pool instead of its merge-check slots (`pool_env`); `--timeout S` is the run's (`research run --timeout`, custody 24 h)."""
    on = base = pool = timeout = None
    agreement = priority = False
    rest = []
    it = iter(argv)
    for x in it:
        if x == "--on":
            on = next(it)
        elif x == "--pool":
            pool = next(it)
        elif x == "--base":
            base = next(it)
        elif x == "--agreement":
            agreement = True
        elif x == "--priority":
            priority = True
        elif x == "--timeout":
            timeout = next(it)
        else:
            rest.append(x)
    try:
        pooled = pool_env(pool, on, priority) + (["--timeout", timeout, "--custody-ttl", "24h"] if timeout else [])
    except ValueError as e:
        print(f"check: --pool {e}", file=sys.stderr)
        return 2
    pair, both = None, []
    if on == "auto" or (on and "," in on):
        names = list(AUTO) if on == "auto" else on.split(",")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        docs, got, why, skipped = place(names, head, priority)
        for s in skipped:
            print(f"check: --on {on}: skipping {s}", flush=True)
        if got is None:
            print(f"check: --on {on} refused: {why} ({', '.join(names)}); nothing launched", file=sys.stderr, flush=True)
            return 2
        print(f"check: --on {on}: {got}: {why}", flush=True)
        on = got
        if both := twins(names, docs, got, priority):
            pair = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.urandom(2).hex()}"
            print(f"check: --on: twins {pair} on {', '.join(both)}: the first to take a check slot runs, and the others leave "
                  f"their line (exit {SLOT.NOT_STARTED}, not started)", flush=True)
    spec = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    combining = rest[:1] == ["--combine"]
    parts = None
    deferred = rest[rest.index("--deferred") + 1] if "--deferred" in rest[:-1] else None
    try:
        if "--shard" in rest[:-1]:
            parts = shard(rest[rest.index("--shard") + 1])["parts"]
        elif deferred is not None:
            parts = deferred_parts(deferred)
    except DECLARATION.DeclarationError as e:
        print(f"check: {e}", file=sys.stderr)
        return 2
    if base is None:
        subprocess.run(["git", "fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main"], cwd=ROOT, capture_output=True)
        base = "origin/main"
    if not landable(base):
        print(f"check: this commit can't land as is; the train's check is the merge check (merging it into {base} gives a tree "
              f"other than its own, so `research merge` won't take this check)")
    changed = [p for p in subprocess.run(["git", "diff", "--no-renames", "--name-only", "-z", base, "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True, errors="surrogateescape", check=True).stdout.split("\0") if p]   # a rename names its source too
    from research.merge import rule_touches

    need = [s for s, rule in tool.CHECK.merge_requires.items() if rule_touches(rule, changed)] + ["lean-agreement"] * agreement
    if combining or parts is not None:                 # a shard gets what its own steps read; a combine runs none
        need = [s for s in need if parts is not None and s in parts]
    # a post-landing run is no check of the commit: recorded without the tool, so `research merge` never reads it.  On a machine
    # every run of a commit shares the shipped tree <root>/src/<sha>, and two checks' Lake builds there broke each other's .lake,
    # so each runs in its own clone of it
    cmd = [sys.executable, "-m", "research", "run", *(["--tool", "check"] if deferred is None else []), "--project", "verity",
           "--cwd", "clone" if on else "source", *(x for p in DECLARED for x in ("--declared-output", p)), *pooled]
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
    audits = not combining and (parts is None or any(p.startswith("lean-audit:") for p in parts))
    if audits and (urls := lean_deps_url_file(d)):
        sent = json.loads(urls.read_text())
        shared = sum(k.startswith("lean-artifacts/") for k in sent)
        print(f"check: sending presigned URLs for {len(sent) - shared} pinned Lean dependency bundle(s) and {shared} Lean artifact "
              f"bundle(s) of recent checks")
        if on:
            cmd += ["--send", str(urls)]
        else:
            env[LEAN_DEPS_URLS_ENV] = str(urls)
    if on and (combining or parts is not None):
        (d / SHARD_PARTS).write_text(json.dumps(parts or []))
        cmd += ["--send", str(d / SHARD_PARTS)]
    if not combining and ("--no-cache" not in rest or "--circuit-cache" in rest):
        given = None
        if "--verdicts-in" in rest[:-1]:
            i = rest.index("--verdicts-in")
            given, rest = Path(rest[i + 1]).resolve(), rest[:i] + rest[i + 2:]
        if pack := gather_verdicts(d, given=given):
            rest += ["--verdicts-in", str(pack)]
    if on:
        cmd += ["--on", on, "--source", "."]
        if "--verdicts-in" in rest:                    # the pack travels with the run; on the machine it's found in the run's inputs/
            cmd += ["--send", str(Path(rest[rest.index("--verdicts-in") + 1]).resolve())]
    try:
        if both:
            procs = []
            for m in both:
                c = list(cmd)
                c[c.index("--on") + 1] = m
                procs.append(subprocess.Popen([*c, "--", *slot_wrap(priority, pair if m == on else f"{pair}@{on}"),
                                               *tool.CHECK.command, *rest], cwd=ROOT, env=env))
            rcs = [p.wait() for p in procs]
            return 0 if 0 in rcs else next((rc for rc in rcs if rc != SLOT.NOT_STARTED), rcs[0])
        cmd += ["--", *(slot_wrap(priority) if on else []), *tool.CHECK.command, *rest]
        return subprocess.run(cmd, cwd=ROOT, env=env).returncode
    finally:
        shutil.rmtree(d, ignore_errors=True)


#: how many of the newest recorded checks' verdict packs a recorded check gathers (`gather_verdicts`)
GATHER = 12


def gather_verdicts(d: Path, store=None, limit: int = GATHER, given: Path | None = None) -> Path | None:
    """One pack (`verdicts.py combine`) of the verdicts the ``limit`` recorded checks that ended last stored, for `--record` to
    send with `--verdicts-in`: a suite whose key one of them earned is reused on any machine, and names the run that earned it.  Only
    recorded runs' packs are taken, a failed or cancelled run's too: its pack holds only the passes its steps stored, each
    under a key of everything it read.  A pack ``given`` (`--record --verdicts-in`) is added to them, never in their place, so no
    flag turns reuse off but `--no-cache`.  A recorded run's pack that `combine` would refuse is passed over by name, so its
    verdicts are recomputed, never reused; a refused pack given is an error.  None when there is no pack at all (the check then
    runs on its own cache)."""
    from research.store.local import LocalStore

    if given is not None and not given.is_file():
        raise SystemExit(f"check: --verdicts-in {given}: no such pack")
    if given is not None and (why := VERDICTS.refusal(given)):
        raise SystemExit(f"check: --verdicts-in {given}: refused ({why})")
    packs = [given] if given is not None else []
    try:
        packs += [p for _, p in gathered_packs(d, store or LocalStore(), limit)]
    except Exception as e:  # noqa: BLE001  (no pack: the check reuses only this machine's passes and the one given)
        print(f"check: no verdicts gathered from the store ({type(e).__name__}: {e})", flush=True)
        del packs[given is not None:]
    if not packs:
        return None
    n = VERDICTS.combine(packs, d / "verdicts-gathered.tar.gz")
    names = [*([f"{given.name} (given)"] if given is not None else []), *(p.parent.name for p in packs[given is not None:])]
    print(f"check: verdicts: {n} entries gathered from {len(packs) - (given is not None)} recorded checks"
          f"{' and the pack given' if given is not None else ''} ({', '.join(names)})", flush=True)
    return d / "verdicts-gathered.tar.gz"


#: the packs a recorded check writes in its run dir, which `record` declares outputs: the passes its steps stored (`Packs`), and
#: those it found before they ran (`Reused`)
REUSED_PACK = "verdicts-reused.tar.xz"
PACK_FILES = ("verdicts.tar.gz", REUSED_PACK)
#: what `record` declares outputs, which custody uploads however the run ends: the packs and the final live record
DECLARED = (*PACK_FILES, LIVE.LIVE_FILE)


def attempt_pack(store, rid: str, d: Path, name: str = "verdicts.tar.gz") -> Path | None:
    """The verdict pack the attempt ``rid`` published (``name`` among its run files, or in its run record when a SIGKILL left it
    no result.json to list them), fetched under ``d``, or None."""
    out = (store.get_attempt(rid) or {}).get("outputs") or {}
    for art in dict.fromkeys(a for a in (out.get("run_files"), out.get("run_record")) if a):
        if any(f["path"] == name for f in store.listing(art)):
            return store.fetch(art, d / name.split(".", 1)[0] / rid, paths=[name]) / name
    return None


def gathered_packs(d: Path, store, limit: int = GATHER) -> list[tuple[str, Path]]:
    """(attempt, pack) of the ``limit`` recorded checks that ended last and published a pack `combine` takes, newest first
    (`gather_verdicts`); one it refuses is passed over by name.  The store's attempts are refreshed first when it has a remote."""
    if store.remote is not None:
        store.refresh(attempts_only=True)
    rows = store.index.sql(f"SELECT id FROM attempts WHERE {COMMON.RECORDED_CHECKS} AND state IN ('done', 'failed', 'cancelled') "
                           "ORDER BY julianday(start_utc) + COALESCE(wall_s, 0) / 86400.0 DESC LIMIT ?", (4 * limit,))
    packs = []
    for (rid,) in rows:
        if (pack := attempt_pack(store, rid, d)) is None:
            continue
        if why := VERDICTS.refusal(pack):
            print(f"check: verdicts: skipping {rid}'s pack, which import refuses ({why}); its verdicts are recomputed", flush=True)
            continue
        packs.append((rid, pack))
        if len(packs) == limit:
            break
    return packs


#: the pack paths of circuit-check's per-target cache (`groups`' `--cache`), all a run with `--no-cache --circuit-cache` imports
CIRCUIT_ENTRIES = "check/circuit-check/"


def verdicts_in(pack: Path, run_dir: Path | None, only: str | None = None) -> dict:
    """Import a verdict pack before the steps run (`verdicts.py`): the path given, else its name in a recorded run's `inputs/`;
    with ``only``, just its entries under that pack path."""
    if not pack.exists() and run_dir is not None and (run_dir / "inputs" / pack.name).exists():
        pack = run_dir / "inputs" / pack.name
    added, kept = VERDICTS.import_(pack, only)
    print(f"check: verdicts: {added} imported, {kept} already here, from {pack.name}{f' (only {only})' if only else ''}", flush=True)
    return {"pack": pack.name, "sha256": _sha256(pack), "added": added, "already": kept, **({"only": only} if only else {})}


#: The suites whose tests run `lake` or what the Lean steps build under `.lake/`: a step of the Lean group once `lean-build` has
#: built the verifier, with elan's `lake` on the PATH, so what they run or skip never depends on how far a concurrent
#: `lake build` has got, and no test's `lake build` shares a build directory with one: lean-audit and lean-changed, which run
#: beside them, build only in trees of their own (tests/test_no_wall_clock.py holds every other suite to not skipping on them).
#: `suites.py` run on its own builds the verifier before them itself; here it is told the build is done (`--lean-built`).
AFTER_LEAN = SUITES.AFTER_LEAN


#: what pytest and the Lean steps hold beside circuit-check in one run, in GiB: in r20261012-155204-6475, pytest and lean-audit
#: held 160 to 350 between them, and circuit-check, given the slot's whole mem=, took the run's scope to its cap (an OOM kill)
SIBLINGS_GB = 350


def cc_mem_gb(slot_gb: int, beside: bool, lean: bool = True) -> int:
    """circuit-check's --mem-gb in a slot of ``slot_gb``: all of it alone (a shard without pytest or Lean), else what
    `SIBLINGS_GB` leaves, but never under a quarter of the slot; beside pytest without ``lean`` (a post-landing run of
    circuit-check and pytest), what pytest's share of that (`tests_gb`, half, the other half lean-suites') leaves."""
    if not beside:
        return slot_gb
    shared = max(slot_gb - SIBLINGS_GB, slot_gb // 4)
    return shared if lean else slot_gb - (slot_gb - shared) // 2


#: where circuit-check finds the memory it is granted once it runs alone in the slot (its ``--mem-gb-file``, `grant_when_alone`)
CC_GRANT = "circuit-check.mem-gb"


def slot_mem_gb() -> int | None:
    return int(m) if (m := os.environ.get(SLOT.SLOT_MEM_ENV, "")).isdigit() else None


def grant_when_alone(path: Path, beside: list[threading.Event], gb: int) -> None:
    """Write ``gb`` (the slot's mem=) to ``path`` once every step beside circuit-check has ended: from then on it runs alone in
    the slot, and the targets it has not started may take all of it (b103: pytest ended 83 minutes into its 125)."""
    for e in beside:
        e.wait()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{gb}\n")
    os.replace(tmp, path)


def groups(out: Path, n: int, cache: bool = True, urls: Path | None = None,
           only: list[str] | None = None, regenerate: bool = True,
           circuit_cache: bool | None = None) -> list[list[tuple[str, list[str] | str, Path]]]:
    """The independent groups, each a sequence of (name, argv or the reason it is skipped / cannot run, cwd); with ``only``, a
    shard's parts (`shard`), just the jobs those name, lean-audit on its packages there, and all the cores to pytest or to
    circuit-check when the other isn't among them.  ``circuit_cache`` (default ``cache``) is whether circuit-check reuses its
    per-target passes."""
    circuit_cache = cache if circuit_cache is None else circuit_cache
    py = sys.executable
    a = max(1, n // 2)
    b = max(1, n - a)
    if only is not None:
        names = {p.partition(":")[0] for p in only}
        a, b = (n if not names & {"flock-circuit-build", "circuit-check", "flock-rust-tests"} else a), (n if "pytest" not in names else b)
    with_lean = only is None or any(x.startswith("lean") for x in names)
    beside = with_lean or "pytest" in names
    suites = [py, str(ROOT / "tools" / "verity" / "check" / "suites.py")]
    slot_gb = slot_mem_gb()
    with_cc = only is None or "circuit-check" in names
    # pytest and lean-suites run at once, each with this share: what circuit-check beside them both leaves, halved
    tests_gb = None if slot_gb is None else (slot_gb - cc_mem_gb(slot_gb, beside) if with_cc else slot_gb) // 2
    tail = ["--timeouts", "--jobs", str(a), *([] if cache else ["--fresh"]), *(["--mem-gb", str(tests_gb)] if tests_gb is not None else []),
            *(["--run", os.environ["RESEARCH_RUN_ID"]] if os.environ.get("RESEARCH_RUN_ID") else [])]
    lean = lean_steps(out, cache, urls, regenerate)
    after_lean: list[str] = []
    if len(lean) > 1:
        after_lean = [x for s in AFTER_LEAN for x in ("--skip", s)]
        lean_suites = [*suites, *AFTER_LEAN, "--lean-built", "--json", str(out / "suites-lean.json"), "--logs", str(out / "suites"), *tail]
        lean.insert(3, ("lean-suites", ["bash", "-c", ELAN + shlex.join(lean_suites)], ROOT))
    pytest = [*suites, "--json", str(out / "suites.json"), "--logs", str(out / "suites"), *after_lean, *tail]
    cc = [py, "-m", "circuit_check", "--all", "-q", "--jobs", str(b), "--out", str(out / "circuit-check.json"),
          *(["--cache", str(cache_dir() / "circuit-check"), "--catalog-units"] if circuit_cache else []),
          *(["--mem-gb", str(cc_mem_gb(slot_gb, beside, with_lean))] if slot_gb is not None else []),
          *(["--mem-gb-file", str(out / CC_GRANT)] if slot_gb is not None and beside else [])]
    build = ["bash", str(ROOT / "verity" / "ml" / "flock" / "check_build.sh")]
    rust = ["env", f"CARGO_BUILD_JOBS={b}", f"RUST_TEST_THREADS={b}", *build, "test"]
    built = {"pytest": ("pytest", pytest, ROOT), "flock-circuit-build": ("flock-circuit-build", build, ROOT),
             "circuit-check": ("circuit-check", cc, ROOT), "flock-rust-tests": ("flock-rust-tests", rust, ROOT)}
    if len(lean) > 1:
        built |= {s[0]: s for s in lean}
        changed: list[str] | str = ([py, str(LEAN_JOBS / "lean_changed.py"), "--cache", str(cache_dir() / "lean-audit"),
                                     "--deps", str(cache_dir() / "lean-deps"), "--tree", str(cache_dir() / "lean-changed")]
                                    if cache else "skipped: --no-cache: lean-audit builds every package cold")
        built["lean-changed"] = ("lean-changed", changed, ROOT)
    else:
        built["lean-build"] = lean[0]                  # no verifier: one skip, named `lean`, where its group starts
    if undeclared := set(built) - {j.name for j in JOBS}:
        raise DECLARATION.DeclarationError(f"check.py builds {', '.join(sorted(undeclared))}, which ci.toml doesn't declare")
    if only is not None:
        if unbuilt := sorted({p.partition(":")[0] for p in only} - set(built)):
            raise DECLARATION.DeclarationError(f"the shard runs {', '.join(unbuilt)}, which check.py doesn't build on this tree")
        pkgs = [p.partition(":")[2] for p in only if p.startswith("lean-audit:")]
        if pkgs:
            name, argv, cwd = built["lean-audit"]
            built["lean-audit"] = (name, [*argv, *(x for p in pkgs for x in ("--package", p))], cwd)
        built = {k: v for k, v in built.items() if k in names and (k != "lean-audit" or pkgs)}
    gs: dict[int, list[tuple[str, list[str] | str, Path]]] = {}
    for j in JOBS:
        if j.name in built:
            gs.setdefault(j.group, []).append(built[j.name])
    return [gs[k] for k in sorted(gs)]


def steps(out: Path, n: int = 1) -> list[tuple[str, list[str] | str, Path]]:
    return [s for g in groups(out, n) for s in g]


#: a shard's result kind, which the tool doesn't produce: no gate takes a shard alone, only a `--combine` check of its set
SHARD_KIND = "check-shard/v1"
#: a post-landing run's result kind (`--deferred`), which no gate takes: drive reads its steps and suites to revert a landing
DEFERRED_KIND = "check-deferred/v1"


def deferred_parts(steps: str) -> list[str]:
    """`--deferred STEP,...`: every part an unsharded check covers that is one of those ci.toml jobs or a job they need or run
    after (`declaration.needed`).  DeclarationError for a name no job has."""
    chosen = DECLARATION.needed(JOBS, [s for s in steps.split(",") if s])
    return [p for p in shard_parts() if p.partition(":")[0] in chosen]


def shard(spec: str) -> dict:
    """`--shard K/N`: {"k", "n", "parts": shard K's parts under ci.toml's plan with N shards, "all": every part an unsharded
    check covers}, on this tree's Lake packages (`declaration.resolve`).  DeclarationError when there is no such plan or shard,
    or the tree can't be listed."""
    m = re.fullmatch(r"(\d+)/(\d+)", spec)
    plans = DECLARATION.plans()
    if not m or int(m.group(2)) not in plans or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise DECLARATION.DeclarationError(f"--shard {spec}: ci.toml's plans are {', '.join(f'K/{n}' for n in sorted(plans))}")
    k, n = int(m.group(1)), int(m.group(2))
    parts_of = _parts_of()
    return {"k": k, "n": n, "parts": DECLARATION.resolve(plans[n], JOBS, parts_of)[k - 1],
            "all": DECLARATION.all_parts(JOBS, parts_of)}


def _parts_of() -> dict[str, list[str]]:
    files = tracked_files(ROOT)
    if files is None:
        raise DECLARATION.DeclarationError("this tree's files can't be listed, so its Lake packages aren't known")
    return {"lean-audit": _lean_audit().packages(files)}


def shard_parts() -> list[str]:
    """Every part an unsharded check covers on this tree (`declaration.all_parts`)."""
    return DECLARATION.all_parts(JOBS, _parts_of())


def _rev(rev: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", "--verify", "-q", rev], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def combine(ids: list[str], run_dir: Path | None) -> int:
    """`--combine RUN...`: whether the shard Attempts ``ids`` are together a passing check of this clean commit, as `research
    merge` judges them (`research.merge.shard_set`); inside a recorded run (`--record --combine`), its result (`check/v1`): their
    steps, each part once, which the gate admits after reading the shards again itself."""
    from research.merge import shard_set
    from research.store.local import LocalStore

    store = LocalStore()
    if store.remote is not None:
        try:
            store.refresh(attempts_only=True)
        except Exception as e:  # noqa: BLE001  (the local catalog still answers; say it may be behind)
            print(f"check: --combine: refresh from the remote failed ({type(e).__name__}: {e}); using the local catalog", flush=True)
    spec = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    commit = _rev("HEAD") or ""
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    done, why = shard_set(store, ROOT, commit, ids, tool.CHECK.name, tuple(tool.CHECK.command))
    if dirty:
        why = "the checkout has uncommitted changes"
    if run_dir is not None:
        doc = {"schema": "research/result/v0.1", "kind": "check/v1", "run_id": os.environ["RESEARCH_RUN_ID"],
               "workload_fingerprint": {"steps": [s["step"] for s in done or []], "shards": ids},
               "validation": {"status": "failed" if why else "passed", "checker": "tools/verity/check/check.py --combine",
                              "detail": why or f"{len(ids)} shards of {commit[:12]}"},
               "steps": done or [], "shards": ids}
        (run_dir / "result.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"check: --combine {' '.join(ids)}: " + (f"refused: {why}" if why else f"a passing check of {commit[:12]} "
                                                    f"({len(done)} steps)"), flush=True)
    return 1 if why else 0


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
ENV_OWN = frozenset({CUSTODY_ENV, STORE_REQUIRED_ENV, REGRESSION_ENV})


def env() -> dict[str, str]:
    """The steps' environment: this process's variables that `ENV_PASS` and `ENV_PASS_PREFIXES` name, less `ENV_OWN`; PYTHONPATH
    entries inside this tree only (a machine's runner puts its shipped `research` first); and a git identity.  Inside a recorded
    run, the store is required, the run's custody dir is passed on and the vLLM reference regression runs (the module docstring's
    "The evidence store")."""
    out = {k: v for k, v in os.environ.items() if (k in ENV_PASS or k.startswith(ENV_PASS_PREFIXES)) and k not in ENV_OWN}
    if os.environ.get("RESEARCH_RUN_ID"):
        out[STORE_REQUIRED_ENV] = "1"
        out[REGRESSION_ENV] = "1"
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
    """What the agreement job reads: the tracked files `agreement_closure` names (verity/ml/flock/verifier/: `ci.py` and the scripts
    it runs, which import only the standard library and each other, `upstream.json` pinning the upstream build,
    the inputs and the job's settings; and the Lean package `flock-verify` is built from, verity/core/service/flock/, its sources and
    toolchain pin), the interpreter and the platform.  None when the tree's files cannot be listed."""
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
    pre = "verity/ml/flock/"
    return [f for f in files if f == pre + "check_build.sh" or f.startswith(pre + "live/")
            or (f.startswith(pre + "flock-") and f.endswith("-b684b12.patch"))]


def rust_toolchain() -> str | None:
    """The Rust toolchain `check_build.sh` pins: its `RUST_TOOLCHAIN` default (check passes no `RUST_TOOLCHAIN` of its own)."""
    p = ROOT / "verity" / "ml" / "flock" / "check_build.sh"
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
    # rustup >= 1.28.1 would install a missing pin here, outside the script's lock
    r = subprocess.run([rustc, "-vV"], env={**e, "PATH": path, "RUSTUP_TOOLCHAIN": toolchain, "RUSTUP_AUTO_INSTALL": "0"},
                       capture_output=True, text=True)
    if r.returncode:
        return None
    h = hashlib.sha256(json.dumps([r.stdout, platform.platform(), _env_key(e)]).encode())
    for f in rust_tests_inputs(files):
        p = ROOT / f
        h.update(f.encode() + b"\0" + (_sha256(p).encode() if p.is_file() else b"-") + b"\0")
    return h.hexdigest()


def lints_key(e: dict[str, str]) -> str | None:
    """What preflight-lints reads: the whole tree (`LINTS` read every file the repository holds), so the commit's tree, with the
    interpreter, the platform and the environment.  None when the tree differs from its commit (untracked files included), or
    git can't say."""
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, capture_output=True, text=True)
    tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, capture_output=True, text=True)
    if st.returncode or st.stdout.strip() or tree.returncode:
        return None
    return hashlib.sha256(json.dumps([tree.stdout.strip(), list(LINTS), sys.version, platform.platform(), _env_key(e)]).encode()).hexdigest()


#: the steps whose pass is kept under a key of everything they read, and reused while it holds
STEP_KEYS = {"lean-agreement": agreement_key, "flock-rust-tests": rust_tests_key, "preflight-lints": lints_key}


def agreement_closure(files: list[str]) -> list[str]:
    """The tracked files the agreement job can read: verity/ml/flock/verifier/, less any Lake package nested in its Lean package
    (C-Flock's proofs are under `Security/Proofs/`), which `flock-verify` is not built from as long as the Lean package
    requires nothing (a `require` in its lakefile keeps them all in), and less that package's `lean-audit.json`, which only
    `tools/verity/lean/check.py` reads (a test fails if the agreement's scripts or the package's lakefile mention it)."""
    pre, lean = "verity/ml/flock/verifier/", "verity/core/service/flock/"
    lakefiles = [f for f in files if f.startswith(lean) and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")]
    root = [f for f in lakefiles if f.count("/") == lean.count("/")]
    nested = tuple(f.rsplit("/", 1)[0] + "/" for f in lakefiles if f not in root)
    if any("require" in (ROOT / f).read_text() for f in root if (ROOT / f).is_file()):
        nested = ()
    return [f for f in files if f.startswith((pre, lean)) and f != lean + "lean-audit.json" and not f.startswith(nested or ("\0",))]


#: the repository-wide lints the preflight runs: each fails a whole check, and together they take seconds
LINTS = ("tests/test_no_wall_clock.py", "tests/test_repository.py")
#: how long a stopped step's processes have after SIGINT before SIGKILL
STOP_GRACE_S = 60
#: each step's record, one JSON line appended as the step ends
STEPS_LOG = "steps.jsonl"


def preflight() -> list[tuple[str, list[str] | str, Path]]:
    """The steps before the groups: `uv lock --check` (a current lock needs no index), then `LINTS` (without bytecode or
    pytest's cache, so they leave nothing in the tree), then `fetchable_fixtures`. Under `uv run`, `$UV` names uv."""
    uv = os.environ.get("UV") or shutil.which("uv")
    lock = [uv, "lock", "--check"] if uv else "failed: no `uv` here to check `uv.lock` (run check through `uv run`)"
    return [("preflight-lock", lock, ROOT),
            ("preflight-lints", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *LINTS], ROOT),
            ("preflight-fixtures", [sys.executable, "-B", str(Path(__file__).resolve()), "--fixtures"], ROOT)]


#: preflight steps that judge the machine the check runs on, so `--preflight` elsewhere (a train's prep) only reports them
ON_NODE = ("preflight-fixtures",)


def fetchable_fixtures(root: Path = ROOT, url: str | None = None, head=None, store=None) -> tuple[dict[str, str], list[str]]:
    """({id: where}, [id]): where each fetched fixture (a `fixtures/artifacts.json` entry whose path git doesn't hold) can come
    from on this machine, and those that can come from nowhere: the public export, asked for its manifest only (one HEAD, no
    blob), else the tree already holding its bytes, else this machine's store holding its manifest and every blob.  The closing
    train's check (cb20, 10 Oct) failed `lean-suites` after its launch on #1716's rec-coins fixture, preserved privately and
    never exported: node 1 fetches from the public export and its own store, and had it in neither.  A copy put in the node's
    store keeps a train moving before the export.  Raises when the public export is needed and there is none."""
    from concurrent.futures import ThreadPoolExecutor

    from research.store.fetchset import public_url
    from research.store.fixtures import _is_tracked, entries, fixture_id, load_map, tracked
    from research.store.local import LocalStore, manifest_key
    from research.store.remote import HttpRemote

    git = tracked(root)
    arts = {a: path for a, path in entries(load_map(root)).items() if not _is_tracked(path, git)}
    ask = sorted(arts)
    if not ask:
        return {}, []
    base = public_url(url)
    if not base:
        raise RuntimeError(f"no public source (RESEARCH_PUBLIC_URL is empty) to ask for {len(ask)} fixture(s)")
    head = head or HttpRemote(base).head
    with ThreadPoolExecutor(max_workers=16) as ex:
        found = list(ex.map(lambda a: head(manifest_key(a.removeprefix("art:"))), ask))
    where = {a: "the public export" for a, h in zip(ask, found) if h is not None}
    where |= {a: "the tree" for a in ask if a not in where and (root / arts[a]).is_file() and fixture_id(root / arts[a]) == a}
    rest = [a for a in ask if a not in where]
    try:
        if rest:
            store = store or LocalStore()
            where |= {a: "this machine's store" for a in rest if store.has_local(a)}
    except Exception as e:                              # noqa: BLE001: a store that can't be read holds nothing
        print(f"check: fixtures: this machine's store can't be read ({type(e).__name__}: {e})", file=sys.stderr)
    return where, [a for a in rest if a not in where]


def fixtures_only() -> int:
    """`--fixtures`: 0 when every fetched fixture can come from somewhere on this machine (`fetchable_fixtures`), saying from
    where; else 1 naming the ids that can't."""
    from collections import Counter

    try:
        where, missing = fetchable_fixtures()
    except Exception as e:                              # noqa: BLE001: a source that can't be asked shows nothing
        print(f"check: fixtures: could not ask the public export ({type(e).__name__}: {e})", file=sys.stderr)
        return 1
    by = Counter(where.values())
    found = ", ".join(f"{n} from {w}" for w, n in sorted(by.items()))
    for a, w in sorted(where.items()):
        if w != "the public export":
            print(f"check: fixtures: {a} from {w}, not the public export")
    if missing:
        print(f"check: fixtures: {len(missing)} fetched fixture(s) registered in fixtures/artifacts.json are in neither this "
              f"machine's store nor the public export ({found or 'none found'}): {' '.join(missing)}; run `research data "
              f"export` for each, or put it in this machine's store", file=sys.stderr)
        return 1
    print(f"check: fixtures: every fetched fixture fixtures/artifacts.json registers can be fetched here ({found or 'none'})")
    return 0


def preflight_only() -> int:
    """`--preflight`: every preflight step, its output here; 0 when all pass but those that judge another machine (`ON_NODE`),
    which only report: a train's prep can't see the node its check runs on, whose own preflight refuses there."""
    failed = []
    for name, argv, cwd in preflight():
        ok = not isinstance(argv, str) and subprocess.run(argv, cwd=cwd).returncode == 0
        if not ok and name in ON_NODE:
            print(f"check: {name}: not refused here: it judges the machine the check runs on, which refuses there", flush=True)
            continue
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


#: a running step none of whose outputs (`outputs`) has changed for this long is hung. pytest sat 23-24 h on one hung test after
#: lean-suites failed, holding its train slot, and circuit-check beside it its share of memory (r20261025-021824-ba9f,
#: r20261025-023312-fa65); a passing step's longest quiet stretch is about an hour (circuit-check's plan before its first target,
#: one 35-minute target, a cold Lean package's build)
IDLE_S = 3 * 3600
#: and one still running after this long is over its budget, whatever it writes (the longest steps passed in 3-4 h)
STEP_BUDGET_S = 12 * 3600
#: how often a running step's outputs are looked at
WATCH_EVERY_S = 60
#: where a step writes as it runs besides the run dir's files named after it: the suites' own logs
WATCH = {"pytest": ("suites",), "lean-suites": ("suites",)}
#: the clock `watch` reads (a test injects one)
CLOCK = time.monotonic


def _exit_within(p: subprocess.Popen, seconds: float) -> int | None:
    try:
        return p.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return None


#: `watch`'s wait for a step's exit code, None while it runs on (a test injects one)
WAIT = _exit_within


def outputs(out: Path, name: str) -> tuple[int, int, int]:
    """What step ``name`` has written so far, as (files, bytes, newest mtime): the run dir's files and directories named after it
    (its log, circuit-check's progress, lean-audit's build logs), and those `WATCH` names for it."""
    roots = [*out.glob(f"{name}*"), *(out / d for d in WATCH.get(name, ()))]
    sizes, newest = [], 0
    for root in roots:
        for f in [root, *(root.rglob("*") if root.is_dir() else ())]:
            try:
                st = f.stat()
            except OSError:
                continue
            sizes.append(st.st_size)
            newest = max(newest, st.st_mtime_ns)
    return len(sizes), sum(sizes), newest


def watch(name: str, p: subprocess.Popen, out: Path) -> tuple[int, str | None]:
    """``p``'s exit code, and why it was stopped: a step quiet for `IDLE_S` or running past `STEP_BUDGET_S` gets SIGTERM (suites.py
    then kills its suites' groups and writes what it has), what is left of it SIGKILL after `STOP_GRACE_S`."""
    start = last = CLOCK()
    seen = outputs(out, name)
    while (rc := WAIT(p, WATCH_EVERY_S)) is None:
        now = CLOCK()
        if (o := outputs(out, name)) != seen:
            seen, last = o, now
        why = (f"hung: no output for {(now - last) / 60:.0f} min" if now - last >= IDLE_S
               else f"over budget: still running after {(now - start) / 3600:.1f} h" if now - start >= STEP_BUDGET_S else None)
        if why:
            print(f"check: {name}: {why}; stopping its processes", flush=True)
            _signal_tree(p.pid, signal.SIGTERM)
            _kill_after_grace(p)
            return p.wait(), why
    return rc, None


#: the steps whose failure stops the run even under --keep-going (`halts` in ci.toml)
HALTS = frozenset(j.name for j in JOBS if j.halts)
#: ... and those whose failure does when their log has this text (`halts_on`); without it (a failed clone or toolchain download)
#: the other groups keep their verdicts
HALTS_ON = {j.name: j.halts_on for j in JOBS if j.halts_on}
#: a step that starts only once the named step in another group has ended (`after` in ci.toml)
AFTER = {j.name: j.after for j in JOBS if j.after}


#: the signal that stopped this check (`_terminated`), and the runs' steps it stops and the packs it has written now
TERMINATED: list[str] = []
_LIVE: "weakref.WeakSet[Steps]" = weakref.WeakSet()
_PACKS: "weakref.WeakSet[Packs]" = weakref.WeakSet()


def _terminated(signum: int, _frame) -> None:
    """SIGTERM's handler: write the run's pack now and stop every run's steps, each from a thread of its own (one may hold
    their lock), so the run ends as stopped and `main` still writes its result and its passes' pack.  A cancel SIGKILLs what is
    left 20 s after its SIGTERM (`research cancel`), which a step's own grace outlasts, so the pack written here is the one its
    custody uploads."""
    TERMINATED.append(name := signal.Signals(signum).name)
    for packs in list(_PACKS):
        threading.Thread(target=packs.request, daemon=True).start()
    for steps in list(_LIVE):
        threading.Thread(target=steps.stop, args=(f"check got {name}", signal.SIGTERM), daemon=True).start()


#: how often a run's pack is written again while its steps run, beside each step's end: pytest's suites store their passes one by
#: one through a step of an hour, and a SIGTERM's write then reads only what they stored since
PACK_EVERY_S = 300


class Packs:
    """A recorded run's verdict pack, `verdicts.tar.gz` in its run dir: what the run's steps stored (`verdicts.export` of what
    changed since ``before``), written again by one thread of its own whenever a step ends, every `PACK_EVERY_S` and on
    SIGTERM (`_terminated`), each time in place of the last (`os.replace`), so a check stopped at any moment leaves the passes it
    had stored by then.  Custody uploads it whatever way the run ends: `record` declares it an output."""

    def __init__(self, out: Path, before: dict[str, str]):
        self.out, self.before = out, before
        self.lock = threading.Lock()
        self.want = threading.Event()
        self.closed = False
        self.written: dict[str, str] | None = None
        self.entries = 0
        self.thread = threading.Thread(target=self._loop, daemon=True, name="verdict-packs")
        self.thread.start()
        _PACKS.add(self)

    def request(self) -> None:
        self.want.set()

    def _loop(self) -> None:
        while not self.closed:
            self.want.wait(PACK_EVERY_S)
            self.want.clear()
            if not self.closed:
                self.write()

    def write(self) -> int:
        """Write the pack unless it holds what is new already; the number of entries it holds."""
        with self.lock:
            try:
                new = VERDICTS.changed(self.before)
                if {k: d for k, (d, _) in new.items()} != self.written:
                    self.entries = VERDICTS.export(self.out, self.before, new)
                    self.written = {k: d for k, (d, _) in new.items()}
            except OSError as e:
                print(f"check: verdicts: writing {self.out.name} failed ({type(e).__name__}: {e}); the last one written stands",
                      flush=True)
            return self.entries

    def close(self) -> int:
        """Stop the thread and write the pack a last time; the number of entries it holds."""
        self.closed = True
        self.want.set()
        self.thread.join()
        return self.write()


#: the reused pack is left out above this, in bytes (custody's line for undeclared files; it is declared, but a pack this big would
#: hold far more than one tree's passes)
REUSED_CAP = 256 << 20
#: the longest its plan may take, in seconds: the check's end waits for it (`Reused.close`), so a plan that hangs packs nothing
REUSED_PLAN_S = 1200


class Reused:
    """The passes a recorded check reuses, packed as `REUSED_PACK` beside `verdicts.tar.gz` (which holds only those its steps
    stored), so one recent check's two packs answer for every part it passed, each entry still naming the run that earned it.
    They are what its plan of its own tree (`plan.py --own`: this run's caches, the gathered pack imported) finds passed, of the
    entries here before its steps ran (``before``).  The plan runs once the twins are built (`run`'s `PREBUILD`, which no other
    process may import while it compiles them), beside the steps; it runs again once circuit-check ends when it found no task
    list to read circuit-check's targets from.  xz with a dictionary spanning many entries (`verdicts.XZ_FILTERS`): one tree's
    circuit-check entries run past a GB, and pack to a few MB.  `gather_verdicts` never takes it; `plan.py` does, for a check
    of the commit it plans and each `--from`.  With ``only`` (`--no-cache --circuit-cache`), just its entries under that pack
    path: the plan also finds suites' passes here, which such a run runs again."""

    def __init__(self, run_dir: Path, before: dict[str, str], only: str | None = None):
        self.out, self.before, self.only = run_dir / REUSED_PACK, before, only
        self.lock = threading.Lock()
        self.circuit_unkeyed = self.circuit_ended = False
        self.entries: int | None = None
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        t = threading.Thread(target=self.write, daemon=True, name="reused-pack")
        self.threads.append(t)
        t.start()

    def ended(self, rec: dict) -> None:
        if rec["step"] == "circuit-check":
            self.circuit_ended = True
            if self.circuit_unkeyed:
                self.start()

    def plan(self) -> dict:
        """`plan.py --tree ROOT --own` in a process of its own (it sets its environment as each step's), under this one's."""
        with tempfile.TemporaryDirectory(prefix="check-reused-") as d:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "plan.py"), "--tree", str(ROOT), "--own",
                                "--out", str(Path(d) / "plan.json")], cwd=ROOT, capture_output=True, text=True,
                               timeout=REUSED_PLAN_S)
            try:
                return json.loads((Path(d) / "plan.json").read_text())
            except (OSError, ValueError):
                raise RuntimeError(f"plan.py --own exited {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}") from None

    def write(self) -> None:
        with self.lock:
            try:
                late = self.circuit_ended
                doc = self.plan()
                self.circuit_unkeyed = any(p["part"] == "circuit-check" and p["state"] == "unkeyed" for p in doc["parts"])
                if self.circuit_unkeyed and self.circuit_ended and not late:
                    self.start()                     # circuit-check ended while this plan ran: its targets are keyed now
                paths = {x for p in doc["parts"] for f in p.get("found") or [] if f.get("source") == "own"
                         for x in f.get("paths") or [] if x in self.before and (self.only is None or x.startswith(self.only))}
                n = VERDICTS.export_paths(self.out, paths, xz=True)
                size = self.out.stat().st_size
                if size > REUSED_CAP:
                    self.out.unlink()
                    print(f"check: verdicts: {REUSED_PACK} left out: {n} entries packed to {size >> 20} MiB, past {REUSED_CAP >> 20} MiB",
                          flush=True)
                    return
                self.entries = n
                print(f"check: verdicts: {n} entries this check reused -> {REUSED_PACK} ({size >> 10} KiB)", flush=True)
            except Exception as e:  # noqa: BLE001  (no reused pack: the passes it reused are cited by their own runs' packs alone)
                print(f"check: verdicts: no {REUSED_PACK} ({type(e).__name__}: {e})", flush=True)

    def close(self) -> int | None:
        while alive := [t for t in self.threads if t.is_alive()]:
            alive[0].join()
        return self.entries


class Steps:
    """The steps of one run: their processes while they run, so that the first failure stops the others (unless ``keep_going``,
    and a step in `HALTS`, or one `HALTS_ON` marks `halts`, stops them even then), and each step's record appended to `STEPS_LOG`
    as it ends, and each start and end to the run's live record (`LIVE`)."""

    def __init__(self, out: Path, keep_going: bool = False, packs: Packs | None = None, reused: Reused | None = None,
                 live=None):
        self.log, self.keep_going, self.packs, self.reused = out / STEPS_LOG, keep_going, packs, reused
        self.live = live or LIVE.Live(out)
        self.lock = threading.RLock()
        self.running: dict[str, subprocess.Popen] = {}
        self.stopped: set[str] = set()
        self.halted: str | None = None
        self.why: str | None = None
        self.log.unlink(missing_ok=True)
        self.spans = SPANS.SpanLog(out)
        _LIVE.add(self)
        if TERMINATED:
            self.stop(f"check got {TERMINATED[0]}")

    @property
    def stopping(self) -> bool:
        return self.why is not None

    def stop(self, why: str, sig: int = signal.SIGINT) -> None:
        """Start nothing more, and stop each running step: ``sig`` to its processes, SIGKILL to what is left after the grace."""
        with self.lock:
            self.why = self.why or why
            for name, p in self.running.items():
                self.stopped.add(name)
                _signal_tree(p.pid, sig)
                threading.Thread(target=_kill_after_grace, args=(p,), daemon=True).start()

    def start(self, name: str, argv: list[str], cwd: Path, env: dict[str, str], log) -> subprocess.Popen | None:
        """The step's process; None, and nothing started, once a failure has stopped the run."""
        with self.lock:
            if self.stopping:
                return None
            self.running[name] = p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.live.started(name)
            return p

    def end(self, rec: dict) -> dict:
        with self.lock:
            self.running.pop(rec["step"], None)
            if rec["step"] in self.stopped and rec["status"] != "passed":
                rec.update(status="stopped", detail=f"stopped: {self.why}")
            rec["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(self.log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.spans.step(rec)
            self.live.ended(rec)
            if self.packs is not None:
                self.packs.request()
            if self.reused is not None:
                self.reused.ended(rec)
            if rec["status"] == "failed" and not self.stopping and (not self.keep_going or rec["step"] in HALTS or rec.get("halts")):
                self.halted = rec["step"]
                self.stop(f"{rec['step']} failed")
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
            rc, hung = watch(name, p, out) if p is not None else (None, None)
        if rc is None:
            rec = {"step": name, "status": "stopped", "detail": f"not started: {steps.why}"}
        elif hung:
            rec = {"step": name, "status": "failed", "rc": rc, "log": log.name, "detail": hung}
        elif rc == -signal.SIGTERM:
            # a cancel (`research cancel`) signals the run's whole process group, so a step can die of it before this check's
            # handler marks it stopped: it found nothing (lean-agreement "failed" with an empty log, r20261026-062546-11fc)
            rec = {"step": name, "status": "stopped", "rc": rc, "log": log.name,
                   "detail": "stopped: killed by a SIGTERM this check did not send (a cancel of the run)"}
        elif rc == 0 and name == "lean-audit" and (why := lean_fast(out)):
            print(f"check: {name}: {why}", flush=True)
            rec = {"step": name, "status": "failed", "rc": rc, "log": log.name, "detail": why}
        else:
            rec = {"step": name, "status": "passed" if rc == 0 else "failed", "rc": rc, "log": log.name}
            if rc and name in HALTS_ON and HALTS_ON[name] in log.read_text(errors="replace"):
                rec["halts"] = True
        if rec["status"] == "passed" and key:
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(json.dumps({"key": key, "argv": argv, "passed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
            rec["cache"] = {"stored": key}
        if name == "circuit-check" and (out / "circuit-check.json").exists():
            c = json.loads((out / "circuit-check.json").read_text()).get("cache")
            # counts only: result.json's steps are its attempt's meta, capped at 256 KiB, and a list per target outgrows it
            rec["cache"] = {k: len(v) if isinstance(v, list) else v for k, v in c.items()} if isinstance(c, dict) else c
    rec["seconds"] = round(time.monotonic() - t0, 1)
    rec = steps.end(rec)
    if rec["status"] == "failed" and log.exists():
        tail = log.read_bytes().decode(errors="replace").splitlines()[-40:]
        print("\n".join(f"  {name}| {line}" for line in tail), flush=True)
    print(f"check: {name}: {rec['status']}{' (cached)' if hit else ''} in {rec['seconds']} s", flush=True)
    return rec


#: vLLM's C++ twin models, compiled on first use: built once before the groups start, so that no step compiles them while
#: another loads them (a check that compiles mid-run is untraceable, and circuit-check stores none of its results)
PREBUILD = ("try:\n    from verity.ml.definitions.numerics import twins\nexcept ImportError:\n    twins = None\n"
            "if twins:\n    twins.preload()\n")


def run(out: Path, n: int | None = None, cache: bool = True, keep_going: bool = False,
        only: list[str] | None = None, packs: Packs | None = None, reused: Reused | None = None,
        regenerate: bool = True, circuit_cache: bool | None = None, live=None) -> tuple[bool, list[dict]]:
    n = n or cores()
    urls = take_lean_deps_urls(out)
    steps = Steps(out, keep_going, packs, reused, live)

    ended: dict[str, threading.Event] = {}
    status: dict[str, str] = {}

    def group(g: list) -> list[dict]:
        done = []
        try:
            for name, argv, cwd in g:
                if (first := AFTER.get(name)) in ended:
                    ended[first].wait()
                    if status.get(first) not in ("passed", "skipped"):
                        done.append(steps.end({"step": name, "status": "stopped", "seconds": 0.0,
                                               "detail": f"not started: {first} {status.get(first, 'did not run')}"}))
                        break
                done.append(_step(name, argv, cwd, out, cache, steps))
                status[name] = done[-1]["status"]
                ended[name].set()
                if done[-1]["status"] in ("failed", "stopped"):
                    break
        finally:
            for name, _, _ in g:
                ended[name].set()
        return done

    try:
        pre = preflight()
        steps.live.plan([name for name, _, _ in pre])
        results = [_step(name, argv, cwd, out, cache, steps) for name, argv, cwd in pre]
        if not steps.stopping and all(r["status"] == "passed" for r in results):
            subprocess.run([sys.executable, "-c", PREBUILD], cwd=ROOT, env=env(), capture_output=True)
            if reused is not None:
                reused.start()
            gs = groups(out, n, cache, urls, only=only, regenerate=regenerate, circuit_cache=circuit_cache)
            steps.live.plan([name for g in gs for name, _, _ in g])
            steps.live.request()
            ended.update({name: threading.Event() for g in gs for name, _, _ in g})
            beside = [ended[name] for g in gs if all(s[0] != "circuit-check" for s in g) for name, _, _ in g]
            grant = None
            (out / CC_GRANT).unlink(missing_ok=True)
            if (gb := slot_mem_gb()) is not None and beside and len(beside) < len(ended):
                grant = threading.Thread(target=grant_when_alone, args=(out / CC_GRANT, beside, gb), daemon=True)
                grant.start()
            with ThreadPoolExecutor(max_workers=max(1, len(gs))) as ex:
                results += [r for rs in ex.map(group, gs) for r in rs]
            if grant is not None:
                grant.join()
    finally:
        if urls and urls.name.startswith("lean-deps-urls-"):
            urls.unlink(missing_ok=True)
    ok = not any(r["status"] in ("failed", "stopped") for r in results)
    steps.spans.close("ok" if ok else ("cancelled" if TERMINATED else "failed"))
    return ok, results


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


def reuse(out: Path, done: list[dict]) -> dict:
    """Per step that can reuse a kept pass, how many of its parts this check reused and how many it ran again: suites
    (`suites.json`, `suites-lean.json`; each suite run again with the inputs that forced it, `suites.forced_by`), circuit-check
    targets (its step's cache), Lean packages (`lean-audit/audit.json`'s `cached`), and the steps cached whole (`STEP_KEYS`)."""
    got: dict[str, dict] = {}
    for f, step in (("suites.json", "pytest"), ("suites-lean.json", "lean-suites")):
        try:
            rows = [r for r in json.loads((out / f).read_text()) if r.get("status") != "skipped"]
        except (OSError, ValueError):
            continue
        ran = sorted((r for r in rows if r.get("status") != "cached"), key=lambda r: -(r.get("seconds") or 0))
        got[step] = {"of": "suites", "reused": len(rows) - len(ran), "rerun": len(ran),
                     "reran": [{"suite": r["suite"], "seconds": r.get("seconds"),
                                "forced_by": ((r.get("forced_by") or {}).get("why") or ["not recorded"])[:3]} for r in ran]}
    for s in done:
        c = s.get("cache") or {}
        if s["step"] == "circuit-check" and "hits" in c:
            misses = {k: c.get(k) or 0 for k in ("untraced", "errors", "changed")}
            got["circuit-check"] = {"of": "targets", "reused": c["hits"], "rerun": c.get("stored", 0) + sum(misses.values()),
                                    "not_stored": misses}
        elif s["step"] in STEP_KEYS and s["status"] in ("passed", "failed"):
            got[s["step"]] = {"of": "step", "reused": int("hit" in c), "rerun": int("hit" not in c)}
        elif s["step"] == "lean-audit" and s["status"] in ("passed", "failed"):
            try:
                a = json.loads((out / "lean-audit" / "audit.json").read_text())
            except (OSError, ValueError):
                continue
            hit = set(a.get("cached") or {})
            every = hit | {p["package"] for p in a.get("packages") or [] if isinstance(p, dict) and "package" in p}
            got["lean-audit"] = {"of": "packages", "reused": len(hit), "rerun": len(every - hit), "reran": sorted(every - hit)}
    for g in got.values():
        g["rate"] = round(g["reused"] / (g["reused"] + g["rerun"]), 3) if g["reused"] + g["rerun"] else None
    return {"summary": ", ".join(f"{step} {g['reused']}/{g['reused'] + g['rerun']} {g['of']}" for step, g in got.items()),
            "steps": got}


def reuse_lines(r: dict) -> list[str]:
    """`reuse` as the log prints it: per step in one line (no rate across steps: thousands of circuit-check targets would hide
    every suite run again), then each step's, then each suite run again with what forced it."""
    lines = [f"check: reused {r['summary'] or 'nothing'}"]
    for step, g in r["steps"].items():
        lines.append(f"check: reuse   {step}: {g['reused']} {g['of']} reused, {g['rerun']} run again")
        for x in g.get("reran") or []:
            if isinstance(x, dict):
                lines.append(f"check: reuse     {x['suite']} ({x['seconds']} s): {'; '.join(x['forced_by'])[:300]}")
    return lines


def write_result(run_dir: Path, ok: bool, done: list[dict], verdicts: dict | None = None, contradicted: list[dict] = (),
                 nondeterministic: list[dict] = (), shard: dict | None = None, stopped_by: str | None = None) -> None:
    """``result.json``: `check/v1`; for a shard (``shard``, from `shard`), `SHARD_KIND`, with the shard's plan, commit and
    config (the tree of `tools/verity/check/` at HEAD) and each of its steps' parts (the plan's names that step ran); for a
    post-landing run (`--deferred`), `DEFERRED_KIND`, the same under `deferred`.  A run
    ``stopped_by`` a signal is `not_run`: no verdict, so a job queue runs it again."""
    if shard is not None:
        done = [{**s, "parts": got} if (got := [p for p in shard["parts"] if p.partition(":")[0] == s["step"]]) else s
                for s in done]
        shard = {**shard, "commit": _rev("HEAD"), "config": _rev("HEAD:tools/verity/check")}
    deferred = bool(shard and shard.get("deferred"))
    kind = DEFERRED_KIND if deferred else SHARD_KIND if shard else "check/v1"
    doc = {"schema": "research/result/v0.1", "kind": kind, "run_id": os.environ["RESEARCH_RUN_ID"],
           "reuse": reuse(run_dir, done), "workload_fingerprint": {"steps": [s["step"] for s in done]},
           "validation": {"status": "not_run" if stopped_by else "passed" if ok else "failed", "checker": "tools/verity/check/check.py",
                          "detail": (f"stopped by {stopped_by}: " if stopped_by else "")
                                    + "; ".join(f"{s['step']} {s['status']}" for s in done)},
           "measurements": [{"name": f"{s['step']}_wall", "value": s["seconds"], "unit": "s"} for s in done],
           "steps": done, "artifacts": [p for p in ("suites.json", "suites-lean.json", "suites/", "circuit-check.json", "lean-audit/",
                                                    *PACK_FILES, STEPS_LOG, SPANS.SPANS_LOG, LIVE.LIVE_FILE,
                                                    *(s.get("log") for s in done), *agreement_verdicts(run_dir))
                                        if p and (run_dir / p).exists()]}
    if shard:
        doc["deferred" if deferred else "shard"] = shard
    if verdicts:
        doc["verdicts"] = verdicts
    if contradicted:
        doc["cache_contradictions"] = list(contradicted)
    if nondeterministic:
        doc["nondeterministic"] = list(nondeterministic)
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def live_record(out: Path, run_dir: Path | None, sharded: dict | None, picked: list[str]):
    """The run's `LIVE.Live`: what it checks (a shard, a post-landing run's steps, parts, or the whole check) of which commit, published
    with the run's custody key when it has one (`custody_dir`)."""
    what = (f"shard {sharded['k']}/{sharded['n']}" if sharded and "k" in sharded else
            f"deferred {','.join(sharded['deferred'])}" if sharded and sharded.get("deferred") else
            f"parts {','.join(picked)}" if picked else "check")
    put, custody = None, custody_dir() if run_dir is not None else None
    if custody is not None:
        try:
            put = LIVE.custody_put(custody)
        except Exception as e:  # noqa: BLE001  (unpublished, the record stays in the run dir)
            print(f"check: live: no publishing ({type(e).__name__}: {e}); {LIVE.LIVE_FILE} stays in the run dir", flush=True)
    run_id = custody.parent.name if custody is not None else run_dir.name if run_dir is not None else None
    return LIVE.Live(out, run_id=run_id, put=put, commit=_rev("HEAD"), what=what)


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
        os.execve(cmd[0], cmd, {**os.environ, "LEAN_SLOT_POOL": "check"})
    if argv == ["--preflight"]:
        return preflight_only()
    if argv == ["--fixtures"]:
        return fixtures_only()
    if argv[:1] == ["--slots"]:
        return slots_view(argv[1:])
    if argv[:1] == ["--plan"]:
        return _load("plan").main(argv[1:], sys.modules[__name__])
    if "--record" in argv:
        if rid := os.environ.get("RESEARCH_RUN_ID"):
            print(f"check: --record inside a recorded run ({rid}): it launches its own `research run`, which places the check "
                  f"and records it as an Attempt of the clean commit, so launch it from your checkout instead, outside `research "
                  f"run` (the launch is light; an agent VM runs it): `{nested_record(argv)}`", file=sys.stderr)
            return 2
        return record([a for a in argv if a != "--record"])
    if os.environ.get("RESEARCH_RUN_ID") and os.environ.get("VERITY_SKIP_STORE") == "1":
        print("check: VERITY_SKIP_STORE=1 inside a recorded run: a recorded check runs the store-backed tests (the module docstring's "
              "\"The evidence store\")", file=sys.stderr)
        return 2
    if argv[:1] == ["--combine"] and len(argv) > 1:
        return combine(argv[1:], Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else None)
    cache = "--no-cache" not in argv
    circuit_cache = cache or "--circuit-cache" in argv
    keep_going = "--keep-going" in argv
    regenerate = "--no-regenerate" not in argv
    rest = [a for a in argv if a not in ("--no-cache", "--circuit-cache", "--keep-going", "--no-regenerate")]
    pack = None
    if "--verdicts-in" in rest and rest.index("--verdicts-in") + 1 < len(rest):
        i = rest.index("--verdicts-in")
        pack, rest = Path(rest[i + 1]), rest[:i] + rest[i + 2:]
    sharded = None
    if "--shard" in rest and rest.index("--shard") + 1 < len(rest):
        i = rest.index("--shard")
        try:
            sharded = shard(rest[i + 1])
        except DECLARATION.DeclarationError as e:
            print(f"check: {e}", file=sys.stderr)
            return 2
        rest = rest[:i] + rest[i + 2:]
        print(f"check: shard {sharded['k']}/{sharded['n']}: {', '.join(sharded['parts'])}", flush=True)
    deferred = None
    if "--deferred" in rest and rest.index("--deferred") + 1 < len(rest):
        i = rest.index("--deferred")
        deferred, rest = rest[i + 1], rest[:i] + rest[i + 2:]
    picked = [a for a in rest if a in DECLARATION.PARTS]
    rest = [a for a in rest if a not in DECLARATION.PARTS]
    if deferred is not None:
        if sharded or picked:
            print("check: --deferred runs the post-landing steps alone, without --shard or a part", file=sys.stderr)
            return 2
        try:
            sharded = {"parts": deferred_parts(deferred), "deferred": deferred.split(",")}
        except DECLARATION.DeclarationError as e:
            print(f"check: {e}", file=sys.stderr)
            return 2
        print(f"check: deferred {deferred}: {', '.join(sorted({p.partition(':')[0] for p in sharded['parts']}))}", flush=True)
    if picked:
        if sharded or os.environ.get("RESEARCH_RUN_ID"):
            print("check: a check of parts runs outside a recorded run and without --shard: no gate takes part of a check",
                  file=sys.stderr)
            return 2
        try:
            chosen = DECLARATION.select(JOBS, picked)
            sharded = {"parts": [p for p in shard_parts() if p.partition(":")[0] in chosen]}
        except DECLARATION.DeclarationError as e:
            print(f"check: {e}", file=sys.stderr)
            return 2
        print(f"check: parts {', '.join(picked)}: {', '.join(j.name for j in JOBS if j.name in chosen)}", flush=True)
    n = None
    if rest[:1] == ["--cores"] and len(rest) == 2 and rest[1].isdigit():
        n, rest = int(rest[1]), []
    if rest:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    if away := SUITES.foreign(SUITES.members(), env()):
        print(f"check: {sys.executable} would run this tree's steps against another checkout's code ({away}); run check with "
              f"{ROOT}'s own environment: `uv run --extra torch-cpu python tools/verity/check/check.py` there", file=sys.stderr)
        return 2
    run_dir = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else None
    out = run_dir or Path(tempfile.mkdtemp(prefix="check-"))
    verdicts = {"in": verdicts_in(pack, run_dir, None if cache else CIRCUIT_ENTRIES)} if pack and circuit_cache else {}
    before = VERDICTS.snapshot() if run_dir is not None and circuit_cache else None
    t0 = time.monotonic()
    TERMINATED.clear()
    main_thread = threading.current_thread() is threading.main_thread()
    prior = signal.signal(signal.SIGTERM, _terminated) if main_thread else None
    packs = Packs(run_dir / "verdicts.tar.gz", before) if before is not None else None
    reused = Reused(run_dir, before, None if cache else CIRCUIT_ENTRIES) if before is not None else None
    live = live_record(out, run_dir, sharded, picked)
    live.start()
    try:
        ok, done = run(out, n, cache, keep_going, sharded["parts"] if sharded else None, packs, reused, regenerate,
                       circuit_cache, live)
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, prior)
        stopped_by = TERMINATED[0] if TERMINATED else None
        TERMINATED.clear()
    ok = ok and not stopped_by
    bad, flaky = contradictions(out), contradictions(out, "nondeterministic")
    for c in bad:
        print(f"check: CACHE CONTRADICTED in {c['suite']}: {', '.join(c['contradicts'][:5])} failed fresh, and a cached pass answers for it",
              flush=True)
    for c in flaky:
        print(f"check: NONDETERMINISTIC in {c['suite']}: {', '.join(c['nondeterministic'][:5])} passed and failed under one key: "
              "delete each (a flaky test is deleted, not rerun or fixed)", flush=True)
    if run_dir is not None:
        if packs is not None:
            verdicts["out"] = {"file": "verdicts.tar.gz", "entries": packs.close()}
            if (got := reused.close()) is not None:
                verdicts["reused"] = {"file": REUSED_PACK, "entries": got}
        write_result(run_dir, ok, done, verdicts, bad, flaky, sharded, stopped_by)
    # after the result: a cancel's SIGKILL follows its SIGTERM in seconds, and the publish may take that long
    live.close("stopped" if stopped_by else "passed" if ok else "failed")
    print("\n".join(reuse_lines(reuse(out, done))), flush=True)
    if stopped_by:
        print(f"check: stopped by {stopped_by} in {time.monotonic() - t0:.1f} s"
              f"{', its passes kept in verdicts.tar.gz' if verdicts.get('out') else ''}; outputs in {out}", flush=True)
        return 128 + signal.Signals[stopped_by].value
    print(f"check: {'passed' if ok else 'FAILED'} in {time.monotonic() - t0:.1f} s ({', '.join(s['step'] + ' ' + s['status'] for s in done)});"
          f" outputs in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
