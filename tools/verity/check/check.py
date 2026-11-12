"""`check`: the repository's one check, what a merge into `main` requires (`research merge`), but for the jobs drive's policy
takes off the merge gate (policy.toml's `off_gate`: circuit-check and ratchet-gate, which run after landing, in drive's
after-landing run on `checked`'s tip).  It reports each step's real verdict; what gates and what may stay red is drive's
(`research.policy`).

    uv run --extra torch-cpu python tools/verity/check/check.py    run every step here, in the checkout, with CPU-only torch
    ... [--keep-going]                                      run every step to its end, even after one has failed
    ... PART...                                             only those parts (`tests`, `circuits`, `lean`: each job's `part` in
                                                            ci.toml) and what their jobs need, after the preflight; outside a
                                                            recorded run, since no gate takes part of a check
    ... --deferred STEP,...                                 only those ci.toml jobs and what they need, after the preflight: the
                                                            after-landing run drive launches on `checked`'s tip (`train.sh
                                                            --deferred`), recorded as `DEFERRED_KIND`, which no gate takes;
                                                            with --shard K/N, only those of shard K's parts, the last
                                                            shard's with those no shard holds (ratchet-gate, where its plan
                                                            doesn't name it) (drive's
                                                            `after_landing_shards`: one run per shard, at once, once the
                                                            records step passed: `--cc-records`)
    ... --no-regenerate                                     lean-audit fails on a lock record difference rather than passing on
                                                            a regenerated record (drive's after-landing run)
    ... --bookkeeping-warn                                  with --deferred and without --no-regenerate (drive's lock run):
                                                            lean-audit repairs a lock's bookkeeping and the caps it breaks and
                                                            warns rather than failing (`lean_audit.py --bookkeeping warn`)
    ... --no-cache [--circuit-cache]                        reuse no pass, build with Lake's local artifact cache off, and run
                                                            every job, those off the gate too (the nightly `audit-main`, which
                                                            gates nothing), and unsharded and without --deferred the opt-in
                                                            tests `OPT_IN` names, with their variables set, after their suites'
                                                            runs; of a pack only the suites' and tests' passes are imported,
                                                            which the fresh suites are compared with; with --circuit-cache,
                                                            circuit-check alone still reuses and stores its per-target passes,
                                                            and its entries of a pack are imported too: drive's after-landing
                                                            run, where a cold `--all` (~4 h) outlasted every run
    uv run python tools/verity/check/check.py --record [--on M]    the same through `research run --tool check` (on machine M, a
                                                            multi-core CPU pod), but for the jobs off the merge gate (`OFF_GATE`,
                                                            policy.toml's `off_gate`: circuit-check, ratchet-gate), which only
                                                            `--deferred` and `--no-cache` run: an Attempt of the clean commit
                                                            checked out, the record `research merge` reads; with presigned URLs
                                                            for the pinned upstream build and inputs when the commit changes
                                                            verity/protocols/flock/ or a Lean package its verifier requires by path
                                                            (but sigma/, a README.md or PROTOCOL.md)
                                                            against --base REF (default origin/main; also
                                                            ratchet-gate's base unless --ratchet-base REF is given), the pinned
                                                            Lean dependency bundles and the newest recorded checks' verdict
                                                            packs (`verdict_url_file`), which the run fetches and the launch
                                                            doesn't; with --on, in a free check slot of M
                                                            (`tools/verity/check/slot.py`), the lock and cores the train checks take,
                                                            and in the run's own clone of the shipped tree (`--cwd clone`);
                                                            --priority takes the next free slot ahead of waiting lane checks,
                                                            and one of TRAIN_GB or more when it runs pytest, lean-suites or
                                                            circuit-check
                                                            (`train_gb`: a machine without one is skipped or refused, exit 75);
                                                            --agreement sends the upstream build's URLs whatever it changes
                                                            (a train that changes a guarantee elsewhere: `train.sh`);
                                                            a warning first when the commit can't land on REF as it is;
                                                            --on auto (the machines in AUTO) or --on M1,M2 picks M
                                                            from `--slots` (`choose`): the first where a check slot no window
                                                            blocks and a Lean `check` slot are free, else the one with the
                                                            fewest waiters ahead (priority ones only, with --priority), and
                                                            refuses when no machine's slots can be read; with no slot free on
                                                            any, twins: the check on each machine it can run on (`twins`),
                                                            the first to take a slot runs; exit 75 when none ran: rerun
    uv run python tools/verity/check/check.py --record --on auto|M --pool NAME [--deferred STEP,...]
                                                            the same in a slot of M's NAME pool (`research run --pool`'s pools),
                                                            not its merge-check slots, which on the CPU nodes take train checks
                                                            alone: a lane's own recorded check.  `--on auto --pool research`
                                                            (any lane's) takes the machine whose research pool `research run
                                                            --on auto` would spill to (`lane_machine`).  Launch it from a
                                                            checkout, never inside a `research run`, where --record is refused
                                                            with the command to run
    uv run python tools/verity/check/check.py --slots [M ...]      each machine's check slots (holders, blocking and upcoming
                                                            windows), check line, and Lean audit slots and their waiters, read
                                                            over ssh by this tree's slot.py and lean_slot.py (`--status --json`);
                                                            then the machine --on auto would take; --json, the documents
    uv run python tools/verity/check/check.py --plan [COMMIT] [--json] [--on MACHINE]
                                                            each part's key at COMMIT (default HEAD) and whether a recorded
                                                            pass under it is here or in a pack a recorded check imports
                                                            (`verdict_packs`);
                                                            runs nothing (`plan.py`), read by `research merge --verdicts`;
                                                            --on: planned on that check node, a short `research run`
    uv run python tools/verity/check/check.py --lean-deps-files    the `--send` for those URLs, for `research merge --train`
    uv run python tools/verity/check/check.py --agreement-files    the `--send` for the upstream build's, likewise
    python tools/verity/check/check.py --preflight                 the preflight alone, for `research merge --train --prepare`
    ... [--verdicts-in PACK]                                import the passes another machine stored (`verdicts.py`) before
                                                            the steps; with --record, beside the newest recorded checks' packs,
                                                            and with --on PACK travels with the run
    uv run python tools/verity/check/check.py --record --shard K/N --on M
                                                            shard K of ci.toml's plan with N shards, on machine M (each shard
                                                            on a pod of the lander's choosing): an Attempt no gate takes alone
    uv run python tools/verity/check/check.py --shard auto:K/N --shard-plan-file PLAN
                                                            shard K of the auto plan PLAN (`shards.py`, `--shard-plan --out`);
                                                            with --record --on M, PLAN goes as the run's input (drive's
                                                            `after_landing_shards = "auto"`, one plan for every shard)
    uv run python tools/verity/check/check.py --shard-plan [--max-shards M] [--n N] [--deferred STEP,...] [--costs FILE]
               [--no-gather] [--split-s S] [--json] [--out FILE]
                                                            the auto plan of this tree, run nothing: its tasks' costs from the
                                                            newest recorded checks (none with --no-gather) and this machine's
                                                            caches, or FILE's {part: seconds} over them
    uv run python tools/verity/check/check.py --record --combine RUN...
                                                            the shards RUN..., one per K, recorded as one check of this commit
                                                            when they are together a passing one (`research merge` reads them)
    uv run python tools/verity/check/check.py --record --cc-records --on M
                                                            circuit-check's records step (`cc_records`): every catalog binding's
                                                            record built once and tarred into the run's $OUT, recorded without
                                                            the tool; drive's sharded after-landing run runs it before its
                                                            shards and gives each its artifact (`--cc-records-from ART@SHA256`:
                                                            unpacked into circuit-check's cache, and the shard refused unless
                                                            its bytes have that sha256 and it was built from this tree)

Sharding.  `check.py --shard K/N` runs the preflight and then only shard K's parts of the plan with N shards that `ci.toml`
declares (`declaration.py`: each job in one shard, `lean-audit` split by Lake package, `circuit-check` into its `shards`, its
targets by the hash of each one's id, `lean-suites` by suite, one that declares `[tool.verity.tests] parts = P` into its P
parts by test id, `lean-suites:SUITE:K/P`, with lean-build again on each shard that holds some of them and not it), so one
commit's check runs on N pods at once and takes about as long as its longest shard. Its result is a `check-shard/v1`
(`SHARD_KIND`) that names the shard, the commit, the config (the tree of `tools/verity/check/` at that commit), the shard's parts and every part an unsharded check covers, and
each step the parts it ran. `--record --combine RUN...` records, on the clean commit, a `check/v1` whose steps are the shards'
parts, each once (the preflight's steps once, passed in every shard), when `research.merge.shard_set` accepts them: the
shards' Attempts are passing runs of the declared command with `--shard K/N` on that clean commit, their results passed with
no step failed or stopped, all of one plan (one N, K = 1..N once each) and one config (the commit's), and their parts are
every part, each once, covering every job the commit's `ci.toml` puts on the gate (a shard skips a job off it, which a plan
may still name: circuit-check and ratchet-gate run after landing alone). `research merge` takes the combined Attempt only
after judging its shards again the same way and finding its steps theirs, so a missing, repeated, failed or mismatched shard
passes nothing. A step skipped in its shard (lean-agreement without the upstream build) is skipped in the combined check too.
When circuit-check is on the gate and its parts ran in more than one shard, the combine also joins their records
(`circuit_join`: each shard's `CC_SHARDS` from its Attempt's run files, through `circuit-check --combine`), and fails unless
that gives the verdict the unsharded `--all` gives: a by-calls root whose callee another shard checked is judged only there.
An auto plan (`--shard auto:K/N`, `shards.py`) has finer parts: each suite of pytest and lean-suites (`pytest:SUITE`,
`lean-suites:SUITE`), or each of the parts by test id one declares, is a part of its own, packed by measured cost, and one
measured past `shards.SPLIT_S` (`--split-s`) is M parts by test file (`pytest:SUITE:K/M`, `suite_splits`), whose partition and timing record the plan carries (`splits`), so
each shard runs `suites.py SUITE:K/M@DIGEST --timings` of the planner's partition; the plan travels with each shard as its run's input
(`AUTO_PLAN`; unrecorded, `--shard-plan-file PLAN`), which refuses one whose parts aren't its tree's, and its result names the plan's digest, floor and predicted wall time
(`shard.plan`), so a combine joins only shards of one plan.  The combine also joins a split suite's parts across shards
(`suite_join`, `suites.combine_parts`), and fails unless every part of one partition passed once; a shard that holds every
part of a suite runs it whole (`suites.py`), and that pass is its verdict.

First the preflight, seconds in all: whether `uv.lock` is current (`uv lock --check`), the repository's wall-clock lint and its
tree's own tests (`LINTS`: markdown caps, report-genre names, secret-like literals, blob and data budgets). The groups start only
once it passes, so a check that one of them fails ends in seconds, not hours.

A step that fails stops the others: each running step's processes get SIGINT (Python code runs its `finally`s, so the Lean audit
removes its scratch tree) and whatever is left of them SIGKILL after `STOP_GRACE_S`, and no group starts another step. Such a
step's status is `stopped`. `--keep-going` lets every group run to its end instead, keeping the verdicts its passing steps store,
except after a failed preflight, which starts no group, or a step whose log has the text ci.toml names for it (`halts_on`,
`HALTS_ON`): a Rust crate that doesn't compile (flock-circuit-build's log says `could not compile`) fails flock-rust-tests
too, so nothing still running is worth the slot.
A step whose outputs (its log, the run dir's files named after it, `WATCH`'s) haven't changed for `IDLE_S`, or that is still
running after `STEP_BUDGET_S`, is stopped the same way (SIGTERM, then SIGKILL) and fails, `hung: no output for N min` or `over
budget`, so a hung test can't hold a check slot for a day; each suite is bounded first by its own `--timeouts`.
The steps stay in this process's group, which a runner's timeout and a manual stop signal as a whole. Each step's record is
appended to `steps.jsonl` in the output directory as the step ends, so a failure shows while the rest still run, and mirrored
as a trace/v1 span line in `spans.jsonl` (`spans.py`: the run opens a root span, each step a child as it starts that its
record completes, the run's end completes the root), the frozen shape the observability watcher reads.  The run's live record
(`live.py`, `check-live.json`) holds every planned step's state, elapsed time and progress, and a recorded run publishes it on the
store's remote with its custody key as each step ends and every few minutes, so `verity ci ls` and `verity ci inspect RUN` read a
running check from anywhere without its node.  A SIGTERM
(`research cancel`, a runner's timeout) stops the steps the same way (a step it reaches first and kills is `stopped` too, not
failed), and a recorded run still writes its result, validation `not_run`, and the pack of the passes its steps stored before
it, then exits 143: a cancelled check's passes are reused.

After the preflight the C++ models are built once (`prebuild`, its output in `prebuild.log`): a prebuild that fails refuses the
run, a failed `prebuild` record and no group started, since each step would otherwise compile them untraced the first time it
ran one.

Then the independent groups run at once, the cores split between the first two (`--cores N`, default the CPUs this process may
use). Which group a step is in, its order there, a step in another group it waits on, and whether its failure halts the run
come from `tools/verity/check/ci.toml` (`declaration.py`); this file builds each step's command, and refuses one the declaration
doesn't name:

    pytest          every test suite (`tools/verity/check/suites.py`) but those that use Lean, half the cores' worth at once: each
                    package's tests in their own process, a pass reused while its inputs are unchanged; each suite's log and the
                    outcomes (`suites.json`) beside the run's result
    circuit-check   first flock-circuit-build, `verity/protocols/flock/prover/check_build.sh`: `cargo check` of flock-circuit in every CPU
                    feature set a pod builds, the proving builds without `seed-injection` among them (a Flock b684b12 checkout,
                    cached under $FLOCK_CHECKOUT), so a merge that breaks the Rust build ends the run in minutes; then
                    `circuit-check --all --jobs` the other half, whose workers start only while their measured peaks
                    (`circuit_check/heavy.json`) fit the memory the run has, at most its check slot's mem= (`--mem-gb`, from
                    slot.py's CHECK_SLOT_MEM_GB) less `SIBLINGS_GB` when pytest or Lean runs beside it, less pytest's share
                    alone when no Lean step does (`cc_mem_gb`), and all of it once every step beside it has ended
                    (`grant_when_alone`): every registered Definition and template
                    (`tools/verity/circuit_check`), or in a shard that holds some of its parts (`circuit-check:K/N`) only
                    those shards of them (`--shard K/N`, records under `CC_SHARDS`),
                    its report beside the run's result (off the gate: a recorded check skips it, and the after-landing
                    run, `--deferred` with every job, runs it); then flock-rust-tests, `check_build.sh test` on that checkout:
                    `cargo test --release` of flock-live's unit tests and of the five upstream crates the patches change
    lean            `lake build` of the Lean verifier (`verity/protocols/flock/verifier`), then its partition check against
                    `verity.circuits.partition` (`unit_cut_agree.py`), then its agreement with upstream's Rust verifier (`ci.py`) on
                    the upstream build stored once and pinned by hash (`verity/protocols/flock/agreement/upstream.json`), which the step
                    fetches when the run was sent its presigned URLs (`agreement_url_file`); skipped by name without them.
                    `research merge` requires it for a change under verity/protocols/flock/ or a Lean package the verifier
                    requires by path (but sigma/, a README.md or PROTOCOL.md; `tool.py`)
    lean-suites     once lean-build has passed, beside the rest of the lean group: the suites that use Lean (`AFTER_LEAN`,
                    outcomes in `suites-lean.json`) on the verifier it built
    lean-audit      at once with the lean group: the audit of every Lake package with a `lean-audit.json` (`lean_audit.py`,
                    which runs `tools/verity/lean/check.py` on scratch trees of its own: axioms, sorry, declared axioms, escapes,
                    files, kernel bypasses, code run while compiling, dependency .olean digests, the guarantees' records, a
                    kernel replay; its reports under `lean-audit/`).  It reads only the source tree and builds nothing in the
                    checkout, so the lean group's steps never wait on it, nor it on them
    ratchet-gate    `adapters/vllm/tools/ratchet_gate.py BASE HEAD`, the gate as BASE has it: no change above BASE (the
                    commit in the run's `check-ratchet-base.txt` or else `check-base.txt`, which `--record --on` sends: its
                    --ratchet-base REF's and its --base REF's; else origin/main)
                    grows the vLLM integration's lint ratchets.  On a train each `Queue: #N` merge is judged against the one
                    below, so its log names the PR whose merge fails (`  #N (COMMIT): failed: ...`), which drive takes as
                    the culprit.  A BASE at or above HEAD fails it, since it would judge nothing (drive's after-landing
                    run sends one below the tip: the tip of the last run whose ratchet-gate gave a verdict, else the last
                    verified `main`); one that names no commit here skips it.  Off the
                    gate: a recorded check skips it, and the after-landing run, `--deferred` with every job, runs it
                    (sharded, in the shard its plan names, else in the last)

Caches (``$CHECK_CACHE``, default ``~/.cache/verity-check``; ``--no-cache`` for a cold run, whose every step also runs with Lake's
local artifact cache off, `suites.UNCACHED_ENV`), each a hit only on identical inputs:

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
    lean-agreement  a pass is reused when everything the job reads is the same (`agreement_key`: verity/protocols/flock/agreement/'s
                    tracked files, which pin the upstream build and inputs, the interpreter, the platform, the environment)
    flock-rust-tests  likewise (`rust_tests_key`: the script, which pins the Flock commit and the Rust toolchain, the patches,
                    the live crate, that toolchain's `rustc -vV`, the platform, the environment), so only a change to those pays
                    for the Rust tests
    preflight-lints likewise (`lints_key`: the commit's tree, which they read whole, the interpreter, the platform, the
                    environment); a tree with uncommitted changes has no key, and runs them
                    The platform is `suites.platform_id` (not the kernel release) and the environment `_env_key` (the
                    switches and the locale, not paths), so a pass is reused on any node for the same inputs, as a suite's is
    ratchet-gate    the gate's own, under `ratchet-gate/` ($RATCHET_GATE_CACHE): each change's verdict, keyed by the gate's bytes
                    and the change's commits (and BASE, for a verdict that read it)

Inside a recorded run it writes `$RESEARCH_RUN_DIR/result.json` (kind `check/v1`): each step's status, seconds and cache use,
with each step's log beside it, and `verdicts.tar.gz`: the passes this run stored, which another machine imports with
`--verdicts-in`, written again as each step ends, every few minutes and on SIGTERM (`Packs`), so a cancelled check's pack holds
what its steps had passed.  `--record` declares it an output, so custody uploads it however the run ends: a cancel's SIGKILL
leaves no result.json, and the pack is then in the attempt's run record alone (`attempt_pack` reads either).  Custody misses it only
where it uploads nothing: `--no-custody-r2` and a run without `--on`, and a run whose runner is killed or whose machine is lost.
`--record` sends presigned URLs of the newest recorded checks' packs, failed and cancelled ones' too, since a pack holds only
passes, up to `GATHER_BYTES` of them, with the packs the Lean audits of ready marks left (`PACKERS`), those of the marks of the PRs
this checkout holds before the newest, and before all of them the packs of the runs `TRAIN_CARRY_FROM` names, held to no
`GATHER_BYTES` (`carried_packs`: the shards a rebuilt candidate of the verified queue replaces) (`verdict_url_file`): the
launcher fetches none, and the run fetches and imports each before its steps (`verdicts_fetched`), beside a `--verdicts-in` pack, never in its place, so every machine reuses every pass a recorded
check earned; each pack imported is named in the result's `verdicts`.  With `--no-cache` it sends both packs of each recorded
check of the commit first (`commit_packs`), and the run imports only the entries it compares with (`COMPARED`, and
circuit-check's under `--circuit-cache`): its fresh suites' `contradicts` then reads every pass the commit was recorded with, not
only those its slot happens to hold.  The agreement's pinned upstream build and inputs go the same way (`agreement_url_file`):
the launcher signs a URL of each and fetches neither, and lean-agreement fetches them as it starts (`lean_agreement_urls`), so a
run whose agreement pass is reused fetches nothing.

The evidence store.  A recorded check runs the store-backed tests (adapters/vllm's `store_io`) on the store's remote, never
skipped: its steps get VERITY_STORE_REQUIRED=1 (a missing remote fails those tests) and, as CHECK_STORE_CUSTODY, the run's custody
dir (`research run --custody-r2`, the default with --on: a delete-free key with GetObject), which `store_io` hands to its store
CLI only.  So `--record --on` must be launched where the parent R2 key is (the Cursor secrets, or `~/.config/verity/r2.env` when it
holds no AWS_SESSION_TOKEN): a temporary credential cannot mint the custody key, and `research run` refuses.  VERITY_SKIP_STORE,
a debug setting (`settings.py`), is refused inside a recorded run.  Outside one, the store-backed tests skip where no remote is
configured.  Under
VERITY_STORE_REQUIRED=1 the suites' guard fails any skip whose reason says "the evidence store is not reachable", whichever helper
fetched (Flock's `verifier/ci.py` hands the custody key to its store CLI too).

The vLLM reference regression.  A recorded run's steps also get VERITY_REGRESSION=1, so adapters/vllm's suite runs
`tests/regression`'s T0 tier against the frozen rows in the store (about four minutes of tests): every check but the manifest
rebuilds, which take an hour and run after each train on main instead (`post_train.py`).  The quick tier before a mark
(`research queue ready`) and suites.py inside a `research run` set it too, so a PR that breaks a row fails there first.  Under VERITY_STORE_REQUIRED=1 a row's
input or contract that doesn't resolve fails there; only a check `fixtures.toml` retires on a row, with its reason, skips.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.client
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: `ci.py`, the agreement job of the Lean verifier against upstream's Rust verifier, and its other scripts
FLOCK_VERIFIER = ROOT / "verity" / "protocols" / "flock" / "agreement"
#: the Lean verifier, the Lake package `flock-verify` is built from (a join of its own, so a layout move that takes
#: the Lean package away from the scripts rewrites it)
FLOCK_LEAN = ROOT / "verity" / "protocols" / "flock" / "verifier"
#: the recording run's custody dir, for `store_io` (not VERITY_-prefixed: the suites' cache keys on those, and the path is per run)
CUSTODY_ENV = "CHECK_STORE_CUSTODY"
STORE_REQUIRED_ENV = "VERITY_STORE_REQUIRED"
#: adapters/vllm's reference regression (`tests/regression`): on in a recorded run, its T0 tier less the manifest rebuilds
REGRESSION_ENV = "VERITY_REGRESSION"
#: the vLLM integration's ratchet gate, which ratchet-gate runs as its base has it, so no change it judges can weaken it
RATCHET_GATE = "adapters/vllm/tools/ratchet_gate.py"


def custody_dir() -> Path | None:
    """The recording run's custody dir (`<root>/requests/<run id>/custody/`, beside `<root>/runs/<run id>/`) when it holds a key."""
    run_dir, run_id = SETTINGS.RESEARCH_RUN_DIR, SETTINGS.RESEARCH_RUN_ID
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


#: every environment variable check reads, declared once (`settings.py`)
SETTINGS = _load("settings")
DECLARATION = _load("declaration")
#: the jobs `tools/verity/check/ci.toml` declares: their groups, order within a group, and what halts the run (`groups`, `HALTS_ON`)
JOBS = DECLARATION.load()


#: the verdict packs (`verdicts.py`)
VERDICTS = _load("verdicts")
#: circuit-check's binding records, named in the packs and carried through the store (`records.py`)
RECORDS = _load("records")
#: the helpers `lean_audit.py` shares (`common.py`), kept out of this file so that its audit key doesn't cover this file
COMMON = _load("common")
#: trace/v1 spans over the run (`spans.py`): `spans.jsonl` beside `steps.jsonl`, the frozen shape the watcher reads
SPANS = _load("spans")
#: the run's live record (`live.py`): `check-live.json` beside `steps.jsonl`, published on the store's remote as the run goes
LIVE = _load("live")
#: the test runner (`suites.py`), for where the steps' Python imports this tree's packages from
SUITES = _load("suites")
#: the shard planner (`shards.py`): auto plans
SHARDS = _load("shards")
cache_dir = COMMON.cache_dir
cores = COMMON.cores
_sha256 = COMMON.sha256
tracked_files = COMMON.tracked_files


ELAN = SUITES.ELAN


#: upstream's verifier builds and the sets' stored inputs, stored once and pinned by hash (`upstream-build.sh`)
UPSTREAM_PIN = FLOCK_VERIFIER / "upstream.json"
#: the pinned upstream build and inputs, each with its size on the store's remote and a presigned URL of its blob
#: (`agreement_url_file`), among a recorded run's inputs, or named by this variable for a local one
AGREEMENT_URLS = "agreement-urls.json"
AGREEMENT_URLS_ENV = "CHECK_AGREEMENT_URLS"
#: what each pinned file `agreement_url_file` lists carries
AGREEMENT_URL_KEYS = {"art": str, "sha256": str, "bytes": int, "url": str}


def agreement_sent(urls: Path) -> dict[str, dict]:
    """The pinned upstream build and inputs that the file ``urls`` (`take_agreement_urls`) lists with their pinned sha256:
    {"upstream": entry, "inputs": entry} for those it lists so, each entry `AGREEMENT_URL_KEYS`.  OSError or ValueError when
    the file can't be read as JSON."""
    doc = json.loads(urls.read_text())
    pin = json.loads(UPSTREAM_PIN.read_text())
    return {k: p for k in ("upstream", "inputs") if isinstance(doc, dict) and isinstance(p := doc.get(k), dict)
            and all(isinstance(p.get(f), t) for f, t in AGREEMENT_URL_KEYS.items()) and p["sha256"] == pin[k]["sha256"]}


#: how long the URLs `agreement_url_file` signs stay valid, in seconds: as long as the longest custody key a run can have
#: (`research run --custody-ttl`), since the run fetches them only when its agreement starts, after its wait for a slot and the
#: Lean build, and the agreement has no other copy to fall back on
AGREEMENT_URL_TTL_S = 24 * 3600


class NotSigned(Exception):
    """Why `agreement_url_file` signed no URLs: the run's agreement could not fetch the pinned files."""


def agreement_url_file(d: Path) -> Path:
    """`AGREEMENT_URLS` in ``d`` (mode 600): the pinned upstream build and inputs (`UPSTREAM_PIN`), each with its size on the
    store's remote (a HEAD) and a GET of its blob that the remote signs for `AGREEMENT_URL_TTL_S`, for the run's agreement step
    to fetch (`lean_agreement_urls`): the launcher signs, and fetches nothing.  `NotSigned` when no remote here signs URLs or
    the remote doesn't hold a pinned file."""
    from research.store.local import object_key
    from research.store.remote import ConfigError, load_remote

    pin = json.loads(UPSTREAM_PIN.read_text())
    try:
        remote = load_remote()
    except ConfigError as e:
        raise NotSigned(f"the store's remote can't be used here ({e})") from None
    if not hasattr(remote, "presign"):
        raise NotSigned("no store remote here signs URLs")
    doc = {}
    for k in ("upstream", "inputs"):
        key = object_key(pin[k]["sha256"])
        try:
            head = remote.head(key)
        except (OSError, ValueError, http.client.HTTPException) as e:     # urllib's errors may name the endpoint: the type only
            raise NotSigned(f"the store's remote can't be read ({type(e).__name__}"
                            f"{f' {e.code}' if isinstance(e, urllib.error.HTTPError) else ''})") from None
        if not isinstance((head or {}).get("bytes"), int):
            raise NotSigned(f"the store's remote doesn't hold the pinned {k} ({pin[k]['art']})")
        doc[k] = {"art": pin[k]["art"], "sha256": pin[k]["sha256"], "bytes": head["bytes"],
                  "url": remote.presign(key, AGREEMENT_URL_TTL_S)}
    p = d / AGREEMENT_URLS
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(doc, f)
    return p


#: the presigned URLs of the pinned Lean dependency bundles (`lean_audit.py`), among a recorded run's inputs, or named by this
#: variable for a local one
LEAN_DEPS_URLS = "lean-deps-urls.json"
LEAN_DEPS_URLS_ENV = "CHECK_LEAN_DEPS_URLS"
#: the newest recorded checks' verdict packs, each with a presigned URL of its blob (`verdict_url_file`), among a recorded run's
#: inputs, or named by this variable for a local one
VERDICT_URLS = "verdicts-urls.json"
VERDICT_URLS_ENV = "CHECK_VERDICT_URLS"
#: the parts a recorded shard runs (none for a combine), among its inputs: the preflight asks only for the Lean dependencies of
#: the packages it audits, since it carries URLs for no others
SHARD_PARTS = "check-shard-parts.json"
#: the commit a `--record --on` check's --base REF names, among its inputs: ratchet-gate judges the merges above it
BASE_FILE = "check-base.txt"
#: the commit a `--record --on` check's --ratchet-base REF names, among its inputs: ratchet-gate's base in place of `BASE_FILE`'s
RATCHET_BASE_FILE = "check-ratchet-base.txt"
#: an auto shard's plan (`shards.py`), among its inputs (unrecorded: `--shard-plan-file`)
AUTO_PLAN = "check-shard-plan.json"
#: what a `--record --on` check runs under on the machine: a free check slot's lock and cores (slot.py), from the shipped tree
SLOT_WRAP = ["python3", "tools/verity/check/slot.py", "--"]
#: the least slot memory a train check that runs pytest, lean-suites or circuit-check takes (slot.py's MIN_MEM_ENV): the train
#: slots' mem=. In a lane slot every parallel suite gets one test worker, the one-process suites share one process and circuit-check
#: runs one target at a time: two to four times a train slot's wall time
TRAIN_GB = 200
#: the steps that need a train slot's memory
TRAIN_STEPS = ("pytest", "lean-suites", "circuit-check")


def train_gb(priority: bool, pool: str | None, parts: list[str] | None, combining: bool) -> int | None:
    """The slot memory a recorded check must have (`TRAIN_GB`): a train's (`--priority`, outside a pool) that runs pytest,
    lean-suites or circuit-check, all of a check (``parts`` None) or a shard or after-landing run holding one; else None."""
    if not priority or pool is not None or combining:
        return None
    return TRAIN_GB if parts is None or any(p.partition(":")[0] in TRAIN_STEPS for p in parts) else None


def slot_wrap(priority: bool = False, twin: str | None = None) -> list[str]:
    """SLOT_WRAP naming who asked (RESEARCH_LANE, else RESEARCH_AGENT_URL), which `slot.py --status` on the machine shows, and
    with --priority for the lander's train checks, --rank from CHECK_SLOT_RANK (drive's after-landing run's and lock run's, so
    they take the next slot ahead of unranked priority checks) and --twin for one of a check's twins."""
    by = SETTINGS.RESEARCH_LANE or SETTINGS.RESEARCH_AGENT_URL
    rank = SETTINGS.CHECK_SLOT_RANK
    return [*SLOT_WRAP[:-1], *(["--priority"] if priority else []),
            *(["--rank", rank] if priority and re.fullmatch(r"\d{1,6}", rank) else []), *(["--by", by] if by else []),
            *(["--twin", twin] if twin else []), SLOT_WRAP[-1]]


def twins(names: list[str], docs: dict[str, dict | str], home: str, priority: bool = False,
          min_mem: int | None = None) -> list[str]:
    """The machines among ``names`` to launch twins of a check on (`slot.py --twin`), ``home`` first: every one a check can run
    on (`placement` refuses none) when ``home``, `choose`'s pick, has no slot free for it; else none, and it runs on ``home``."""
    def runs(name: str) -> bool:
        try:
            placement(docs[name], priority, min_mem)
        except (ValueError, KeyError, TypeError, AttributeError):
            return False
        return True

    if placement(docs[home], priority, min_mem)[0]:
        return []
    out = [home, *(n for n in names if n != home and runs(n))]
    return out if len(out) > 1 else []


#: the machines `--on auto` and `--slots` consider, in this order: the shared nodes' merge-check slots (the CPU nodes' are
#: only=trains)
AUTO = ("vy-nebius-2",)
#: the pool a lane's own recorded check takes when it names none (`nested_record`): any lane's, research.auto's SPILL_POOL, on
#: the machine `lane_machine` picks
LANE_POOL = "research"
#: how long reading one machine's slots may take, ssh included
SLOTS_TIMEOUT_S = 60
#: a full check's length: `--on auto` places none on a machine whose deadline (/etc/research/deadline) is nearer than this
FULL_CHECK_S = 9000
#: a lane's own test run, which no landing needs (a PR lands on review, and drive's full check decides what reaches `main`)
LANE_RUN = ('`research run --on auto --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && '
            'uv run tools/verity/check/suites.py --changed $(git fetch -q origin main && git merge-base HEAD FETCH_HEAD) --quick"`')
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


def placement(doc: dict, priority: bool = False, min_mem: int | None = None) -> tuple[bool, int, str]:
    """A check's prospects on a machine whose `read_slots` documents are ``doc``: whether a check slot and a Lean `check` slot
    are free for it (more free than live waiters ahead of it), how many live waiters are ahead of it in the check line (with
    ``priority``, a train check's, only the priority ones), and the state in a few words; with ``min_mem`` (`train_gb`), only
    the slots whose mem= holds it count. ValueError when no check can run there: a document has an error, the slots file lists
    no slot (or none that holds ``min_mem``), the Lean slots file grants check's lean-audit step none, or the machine's deadline
    is nearer than FULL_CHECK_S."""
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
            raise ValueError(f"every slot in {check['file']} is for train checks (only=trains). A PR lands on review, and drive's "
                             "full check decides what reaches `main`, so no lane check is required; to run its tests anyway: " + LANE_RUN)
        if min_mem and not (mine := [s for s in mine if s.get("mem") is None or s["mem"] >= min_mem]):
            raise ValueError(f"no slot in {check['file']} holds this check's {min_mem} GB (" + ", ".join(
                f"{s['name']} {s.get('mem')} GB" for s in check["slots"]) + "): it takes a train slot")
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


def choose(names: list[str], docs: dict[str, dict | str], priority: bool = False,
           min_mem: int | None = None) -> tuple[str | None, str, list[str]]:
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
            room, ahead, why = placement(doc, priority, min_mem)
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
#: checks placed at once go to different machines (without it, parallel launches all read one machine's one free slot as
#: theirs, and queued there while another machine's slot stood free)
PLACED = Path(tempfile.gettempdir()) / "verity-check-placed.json"
PLACED_S = 1800


def _trees(doc: dict) -> set[str]:
    """The shortened commits a machine's check slots hold or wait for: each one's commit, else its tree's name (<root>/src/<sha>,
    from a slot.py that records no commit)."""
    check = doc.get("check") or {}
    who = [s["holder"] for s in check.get("slots") or [] if s.get("holder")] + list(check.get("waiters") or [])
    return {(h.get("commit") or Path(h["tree"]).name)[:12] for h in who if h.get("commit") or h.get("tree")}


def place(names: list[str], commit: str, priority: bool = False, read=None, path: Path | None = None,
          now=time.time, min_mem: int | None = None) -> tuple[dict[str, dict | str], str | None, str, list[str]]:
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
        got, why, skipped = choose(names, docs, priority, min_mem)
        if got:
            placed[commit] = {"on": got, "at": t}
            path.write_text(json.dumps(placed))
    return docs, got, why, skipped


def _who(h: dict, held: bool = False) -> str:
    """A waiter in a few words, or with ``held`` a slot's holder: since when (a holder: `slot.held_times`, the time it has held
    the slot, never its process's age), its run, its tree (a commit's, shortened) and who asked (a URL's last part)."""
    when = SLOT.held_times(h) if held else f"since {h['since']} ({h['minutes']} min)" if h.get("since") else ""
    parts = [when or f"pid {h['pid']}"]
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
            out.append(f"  check {s['name']}  cores {s['cpus']}  " + (f"held {_who(s['holder'], held=True)}" if s["holder"] else "free")
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
            held = [f"{pool}/{s['index']} held {_who(s['holder'], held=True)}" for s in p["slots"] if s["holder"]]
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
    return _take(out, LEAN_DEPS_URLS, LEAN_DEPS_URLS_ENV)


def take_agreement_urls(out: Path) -> Path | None:
    """`AGREEMENT_URLS` sent with the run, taken out of its run directory as `take_lean_deps_urls` takes its file."""
    return _take(out, AGREEMENT_URLS, AGREEMENT_URLS_ENV)


#: what a file `_take` moved out of a run directory is named, beside a random part: `run` deletes it when the run ends
TAKEN = tuple(f"{Path(n).stem}-" for n in (LEAN_DEPS_URLS, AGREEMENT_URLS))


def _take(out: Path, name: str, env_name: str) -> Path | None:
    sent = out / "inputs" / name
    if not sent.is_file():
        return Path(u) if (u := SETTINGS.get(env_name)) else None
    fd, tmp = tempfile.mkstemp(prefix=f"{Path(name).stem}-", suffix=".json")
    with os.fdopen(fd, "wb") as f:
        f.write(sent.read_bytes())
    sent.unlink()
    return Path(tmp)


#: what each pack `verdict_url_file` lists carries
VERDICT_URL_KEYS = {"run": str, "sha256": str, "bytes": int, "url": str}


def take_verdict_urls(out: Path) -> list[dict]:
    """The packs `verdict_url_file` listed for the run, read from its inputs and deleted there before any step starts, since each
    URL grants a read until it expires (custody deletes a file left there: `research.store.custody.CREDENTIALS`); for a local
    run, read from the file `VERDICT_URLS_ENV` names.  [] when the run was sent none, or none that reads as a list of packs."""
    sent = out / "inputs" / VERDICT_URLS
    path = sent if sent.is_file() else Path(u) if (u := SETTINGS.get(VERDICT_URLS_ENV)) else None
    if path is None:
        return []
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"check: verdicts: the packs' URLs sent can't be read ({type(e).__name__}); the run imports none", flush=True)
        doc = []
    finally:
        sent.unlink(missing_ok=True)
    return [p for p in doc if isinstance(p, dict) and all(isinstance(p.get(k), t) for k, t in VERDICT_URL_KEYS.items())] \
        if isinstance(doc, list) else []


#: Lean's job code: the lean-audit step's (`lean_audit.py`), and the fast path's (`lean_changed.py`), whose kind and keys
#: `lean_fast` and `lean_passes` read
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
    and its report names the regenerated lock, which the verified queue's loop commits after a candidate's records run
    (`research.vqueue_records.commit`).
    Read from the job, so a tree from before it runs as it always did."""
    try:
        return '"--regenerate"' in (LEAN_JOBS / "lean_audit.py").read_text()
    except OSError:
        return False


def lean_audit_misses(cache: bool = True, only: list[str] | None = None) -> bool:
    """Whether the lean-audit step will audit a package (of ``only``, its `--package`s, when given): one whose pass isn't kept
    here under its key with what its reuse needs (`lean_audit.reusable`, `lean_audit.run`'s test)."""
    la = _lean_audit()
    files = tracked_files(ROOT)
    if not cache or files is None:
        return True
    pkgs = [p for p in la.packages(files) if not only or p in only]
    return any(not la.reusable(cache_dir() / "lean-audit", p, la.key(p, files, la.FLAGS)) for p in pkgs)


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


def lean_passes(out: Path, done: list[dict], lc=None) -> dict[str, str]:
    """{inputs key: package} for each package the lean-audit step audited and passed on this tree's own lock (no failure, the
    kernel replay, not passed on a record it regenerated), each key `lean_changed.inputs_key` (``lc``, the module) of this tree:
    the key a records run of the commit keeps and `lean_changed.py --inputs-key` prints, so `research queue`'s records gate
    finds a check's passes as it finds a records run's.  None from a step that didn't end with the audit's verdict
    (refused, hung, stopped) or from an audit that failed without reporting a package (check.py crashed)."""
    step = next((s for s in done if s.get("step") == "lean-audit"), None)
    if step is None or step.get("status") not in ("passed", "failed") or step.get("rc") not in (0, 1) or step.get("detail"):
        return {}
    try:
        doc = json.loads((out / "lean-audit" / "audit.json").read_text())
        reports = {r["package"]: r for r in doc["packages"] if isinstance(r.get("package"), str)}
        if any(f not in reports for f in doc.get("failed") or []):
            return {}
        regenerated = doc.get("regenerated") or {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"check: lean-audit's report can't be read ({type(e).__name__}: {e}), so the result lists no Lean passes",
              file=sys.stderr)
        return {}
    lc = lc or _load("lean_changed", LEAN_JOBS)
    ok = [p for p, r in sorted(reports.items()) if r.get("failures") == [] and not r.get("settled") and p not in regenerated
          and r.get("kind") != lc.FAST_KIND and isinstance(r.get("seconds"), dict) and "replay" in r["seconds"]]
    files = tracked_files(ROOT) if ok else None
    return {lc.inputs_key(p, files): p for p in ok} if files is not None else {}


def lean_slot(cmd: list[str]) -> list[str]:
    """`--lean-slot CMD...`, the lean-audit step: CMD in one of the node's Lean audit slots when it will audit a package, as it
    is when it only reuses passes, which would otherwise wait behind an hour-long audit for seconds of work. CMD runs with
    LEAN_SLOT_POOL=check, so an audit it runs after all (a pass gone since) takes its slot from the same pool
    (`tools/verity/lean/check.py --build`)."""
    only = [cmd[i + 1] for i, x in enumerate(cmd[:-1]) if x == "--package"]
    return [*LEAN_SLOT_WRAP, *cmd] if lean_audit_misses("--fresh" not in cmd, only) else cmd


def required_packages(lean: Path) -> dict[str, Path | None]:
    """{name: directory} of the packages the Lean package ``lean`` requires, by its manifest (which lists those its
    requirements require too) or its lakefile: one required by path at that path, a fetched one None."""
    import tomllib

    m, lf = lean / "lake-manifest.json", lean / "lakefile.toml"
    entries = [*(json.loads(m.read_text()).get("packages", []) if m.exists() else []),
               *(tomllib.loads(lf.read_text()).get("require", []) if lf.exists() else [])]
    return {e.get("name", "?"): None if (p := e.get("dir") or e.get("path")) is None else (lean / p).resolve() for e in entries}


def requires_outside(lean: Path) -> list[str]:
    """The packages the Lean package ``lean`` requires from outside the tree, by its manifest or its lakefile: a fetched
    one, or one at a path that leaves the repository. The agreement's pass is kept under the files of the verifier's
    package and of the packages of the tree it requires by path (`agreement_closure`), so it would outlive a change to
    one of these."""
    inside = (lean.resolve(), ROOT.resolve())
    return sorted(n for n, d in required_packages(lean).items() if d is None or not any(d.is_relative_to(r) for r in inside))


def required_in_tree(lean: str) -> tuple[str, ...]:
    """The packages of the tree the Lean package at ``lean`` (from the root, ending in "/") requires by path from outside
    its directory, each as ``lean`` is written."""
    root, own = ROOT.resolve(), (ROOT / lean).resolve()
    return tuple(sorted(d.relative_to(root).as_posix() + "/" for d in required_packages(ROOT / lean).values()
                        if d is not None and d.is_relative_to(root) and not d.is_relative_to(own)))


def lean_steps(out: Path, cache: bool = True, urls: Path | None = None,
               regenerate: bool = True, bookkeeping: bool = False,
               agreement_urls: Path | None = None) -> list[tuple[str, list[str] | str, Path]]:
    """The Lean steps: nothing on a tree without the verifier (a skip, named); then `lake build`, the Lean partition check's
    conformance with `verity.circuits.partition` (`unit_cut_agree.py`), the audit of every Lean package, and the agreement job
    against upstream's Rust verifier on the pinned upstream build (`upstream.json`) when the run was sent its URLs
    (``agreement_urls``, `take_agreement_urls`), which the step fetches; without them that step is skipped by name, and it
    fails when the file sent can't be read or doesn't list both.  The audit regenerates lock records (`lean_regenerates`) unless ``regenerate`` is off (`--no-regenerate`,
    drive's after-landing run, where a record difference fails), with ``bookkeeping`` (`--bookkeeping-warn`, drive's lock run)
    repairs the bookkeeping too and warns (`lean_audit.py --bookkeeping warn`), and inside a recorded run it bundles the module versions it
    built for other nodes (`lean_audit.py --bundle`).  `research merge` requires it for a change under `verity/protocols/flock/` or a Lean package the verifier
    requires by path (`tools/verity/check/tool.py`), and
    `--record` sends its URLs then.  The agreement fails, sent or not, while the verifier's package requires one from outside the
    tree (`requires_outside`)."""
    lean = FLOCK_LEAN
    if not (lean / "lakefile.toml").exists():
        if FLOCK_VERIFIER.is_dir():
            return [("lean", f"the verifier is on this tree but its Lean package is not at {lean}", ROOT)]
        return [("lean", "skipped: verity/protocols/flock/verifier is not on this tree", ROOT)]
    verify = lean / ".lake" / "build" / "bin" / "flock-verify"
    agreement: list[str] | str = ("skipped: the pinned upstream build was not sent (a merge changing verity/protocols/flock/ or "
                                  "a Lean package the verifier requires by path (but sigma/, a README.md or PROTOCOL.md) requires this step: "
                                  "`check.py --record` sends its URLs)")
    if UPSTREAM_PIN.exists() and agreement_urls is not None:
        try:
            got = agreement_sent(agreement_urls)
        except (OSError, ValueError) as e:
            agreement = f"the URLs sent for the pinned upstream build and inputs can't be read ({type(e).__name__})"
        else:
            agreement = ([sys.executable, str(Path(__file__).resolve()), "--lean-agreement-urls", str(out), str(agreement_urls)]
                         if len(got) == 2 else
                         f"the URLs sent list {' and '.join(got) or 'neither'} of the pinned upstream build and inputs")
    if outside := requires_outside(lean):
        agreement = (f"the verifier's Lean package requires {', '.join(outside)} from outside the tree, which the "
                     f"agreement's key (`agreement_closure`) and `research merge`'s lean-agreement rule don't cover")
    return [("lean-build", ["bash", "-c", SUITES.LAKE_BUILD], lean),
            ("lean-unit-cut", [sys.executable, str(FLOCK_VERIFIER / "unit_cut_agree.py"), "--lean", str(verify),
                               "--out", str(out / "lean-unit-cut")], ROOT),
            ("lean-audit", [sys.executable, str(Path(__file__).resolve()), "--lean-slot",
                            sys.executable, str(LEAN_JOBS / "lean_audit.py"), "--out", str(out / "lean-audit"),
                            "--cache", str(cache_dir() / "lean-audit"), "--deps", str(cache_dir() / "lean-deps"),
                            *([] if cache else ["--fresh"]), *(["--urls", str(urls)] if urls else []),
                            *(["--bundle"] if SETTINGS.RESEARCH_RUN_ID else []),
                            *(["--regenerate"] if regenerate and lean_regenerates() else []),
                            *(["--bookkeeping", "warn"] if bookkeeping and regenerate and lean_regenerates() else [])], ROOT),
            ("lean-agreement", agreement, ROOT)]


def memory_gb(root: Path = Path("/")) -> int:
    """The memory this process may use: the least of MemAvailable and what each cgroup limit over it leaves, v2 or v1 (a pod's
/proc/meminfo is the host's; on a VM the limits are on its own cgroup's ancestors, such as a run's session scope). A v2
cgroup's page cache (active_file and inactive_file in its memory.stat) is in its memory.current, but the kernel reclaims it at
the limit, so it counts as left, as in slot.py's `memory_gb`: a user slice can hold hundreds of GB of cache near its limit,
and counted as used it leaves a step refusing to run with "0 GB"."""
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
    (its pinned commit, the command `agree.py` would run, upstream's expected verdict), over the unpacked build and inputs in
    ``w``. Upstream's verdict is the case's `upstream_accepted` where it records an intended divergence (`agree.py`), as set
    16's honest session does: Lean refuses it for completeness, upstream accepts it."""
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
        e = s["sessions"][0]["expect"]
        out[s["pr83"]] = (s["pr83"], [*agree.upstream_cmd(a, str(public), extra), str(honest)],
                          {"accepted": e.get("upstream_accepted", e["accepted"])})
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
                f"machine, or rebuild on this glibc (verity/protocols/flock/agreement/upstream-build.sh) and re-pin")
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


def lean_agreement(out: Path, upstream: Path, inputs: Path, verify: Path | None = None) -> int:
    """Unpack the pinned upstream build and inputs, check every binary against the pin and that this machine can run them, and
    run `ci.py` on every replayable set with ``verify`` (default: this tree's build of flock-verify)."""
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
    argv = [sys.executable, str(FLOCK_VERIFIER / "ci.py"), "--lean", str(verify or FLOCK_LEAN / ".lake" / "build" / "bin" / "flock-verify"),
            *ups, "--work", str(w / "work"), "--inputs", str(w / "inputs"), "--jobs", str(max(1, min(cores(), len(sets)))),
            "--session-jobs", str(session_jobs()),
            "--seed", str(ci.get("seed", 20260926)), "--fuzz", str(ci.get("fuzz", 20)), "--out", str(w)]
    print(f"check: lean-agreement: preflight passed; ci.py on {len(sets)} sets (it prints its results when every set ends)",
          flush=True)
    if rc := subprocess.run(argv, cwd=ROOT).returncode:
        print(f"check: lean-agreement: error: ci.py exited {rc}; its verdicts are in {w}", file=sys.stderr, flush=True)
    return rc


#: the waits before each further fetch of a pinned file the agreement could not fetch, in seconds: it has no other copy to fall
#: back on, so one failed read of the store would fail the step
AGREEMENT_FETCH_WAITS_S = (15, 60)


def lean_agreement_urls(out: Path, urls: Path, sleep=time.sleep) -> int:
    """`--lean-agreement-urls OUT URLS`, the agreement step of a run sent `AGREEMENT_URLS`: fetch the pinned upstream build and
    inputs through their URLs into a scratch directory outside the run directory, each held to its listed size and pinned
    sha256 (`fetch_signed`) and fetched again after each of `AGREEMENT_FETCH_WAITS_S` while it fails, then `lean_agreement` on
    them.  A cached pass of the step runs none of it, so a check that reuses the agreement fetches nothing."""
    try:
        got = agreement_sent(urls)
    except (OSError, ValueError) as e:
        print(f"check: lean-agreement: error: {urls} can't be read ({type(e).__name__})", file=sys.stderr)
        return 1
    if len(got) != 2:
        print(f"check: lean-agreement: error: {urls} lists {' and '.join(got) or 'neither'} of the pinned upstream build and "
              f"inputs", file=sys.stderr)
        return 1
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="check-agreement-") as d:
        files = {k: Path(d) / f"{k}.tar.gz" for k in got}
        for k, p in got.items():
            what = f"the pinned {'upstream build' if k == 'upstream' else k} ({p['art']})"
            for wait in (*AGREEMENT_FETCH_WAITS_S, None):
                if (why := fetch_signed(p, files[k])) is None:
                    break
                if wait is None:
                    print(f"check: lean-agreement: error: {what}: {why} ({len(AGREEMENT_FETCH_WAITS_S) + 1} tries)",
                          file=sys.stderr, flush=True)
                    return 1
                print(f"check: lean-agreement: {what}: {why}; fetching it again in {wait} s", flush=True)
                sleep(wait)
        print(f"check: lean-agreement: fetched the pinned upstream build and inputs ({sum(p['bytes'] for p in got.values()) >> 20} "
              f"MiB) in {time.monotonic() - t0:.0f} s", flush=True)
        return lean_agreement(out, files["upstream"], files["inputs"])


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
    pool `research run --pool NAME` takes (`slot_only.pool_slots`), rather than its merge-check slots, which on the CPU nodes are
    only=trains: slot.py reads the pool's slots file from CHECK_SLOTS.  The command stays `slot_wrap`'s around the declared one,
    not `slot_only.in_pool`'s (`env ... nice -n 19`), so `research merge` and `verdict_packs` read the attempt as a check of its
    commit.  ValueError when the pool can't be taken so; with --on auto or none, `lane_machine` picks the machine."""
    if pool is None:
        return []
    from research.pods.nebius import slot_only

    if not slot_only.POOL_NAME.fullmatch(pool):
        raise ValueError(f"{pool}: no such pool; a pool is a name of letters, digits, '_' or '-' (the pools every node may have are "
                         f"{', '.join(sorted(slot_only.POOLS))}; a node's own are under {slot_only.LOCKS})")
    if on and "," in on:
        raise ValueError(f"{pool} waits in one machine's pool: `--on auto --pool {LANE_POOL}` or --on MACHINE, not a list, which "
                         f"chooses among merge-check slots")
    if priority:
        raise ValueError(f"{pool} takes no --priority: that is the lander's train checks' place in the merge-check slots' line")
    return ["--env", f"CHECK_SLOTS={slot_only.pool_slots(pool)}"]


def lane_machine(pool: str) -> str:
    """The machine a `--record --on auto --pool NAME` check waits on: the one whose NAME pool `research run --on auto` would
    spill a run to now (`research.auto.choose` on the CPU nodes' lines, read as it reads them).  ValueError when NAME is
    not its pool (a node's own pool takes --on MACHINE) or no machine's pool can take it."""
    from research import auto

    if pool != auto.SPILL_POOL:
        raise ValueError(f"{pool}: --on auto takes only research.auto's pools ({auto.SPILL_POOL}); a node's own pool takes "
                         f"--on MACHINE, the machine that keeps it")
    try:
        target, why = auto.choose(auto.read(ROOT, 1), 1, 0)
    except auto.Unplaceable as e:
        raise ValueError(f"{pool}: --on auto: {e}") from None
    if target is None or target[1] != pool:
        raise ValueError(f"{pool}: --on auto: no machine's {pool} pool can take it ({why}); name one with --on MACHINE")
    print(f"check: --on auto --pool {pool}: {target[0]} ({why})", flush=True)
    return target[0]


def nested_record(argv: list[str]) -> str:
    """The command that records ``argv``'s check from outside a `research run`: `record` launches a `research run` of its own,
    so inside one it is refused with this.  With no --on, `--on auto` in its --pool, else `LANE_POOL`."""
    where = [] if "--on" in argv else ["--on", "auto", *([] if "--pool" in argv else ["--pool", LANE_POOL])]
    return shlex.join(["uv", "run", "python", "tools/verity/check/check.py", "--record", *where,
                       *(a for a in argv if a != "--record")])


def record(argv: list[str]) -> int:
    """`--record [--on MACHINE|auto|M1,M2] [--base REF] [--ratchet-base RREF] [--agreement]`: `research run --tool check` on this
    clean commit, with presigned URLs of the pinned upstream build and inputs (`agreement_url_file`: the launch fetches neither)
    when the commit changes a path `tools/verity/check/tool.py` names against REF (default origin/main, fetched first), or
    whenever --agreement is given, refused when they can't be signed, with
    `--on` REF's commit (`BASE_FILE`, ratchet-gate's base unless RREF's commit, `RATCHET_BASE_FILE`, is), and the presigned URLs of
    the newest recorded checks' verdict packs (`verdict_url_file`), with --no-cache after both packs of each recorded check of the
    commit (`commit_packs`, the passes its fresh suites are compared with). A commit that can't land on REF as it is
    (`landable`) gets a warning and is checked anyway. `--on auto` or a list takes the machine `choose` picks, first of all, or
    launches `twins` there and on the others (one `research run` each, waited on together: 0 when one passed, the exit of the
    one that ran otherwise, and slot.py's NOT_STARTED (75) when none ran, say with HOME unreachable: rerun it).  With
    `--deferred STEP,...` it is a `research run` without `--tool check` of those steps alone (no gate reads it), with the
    verdict packs' URLs and no upstream build or Lean dependencies.  `--pool NAME` (with `--on MACHINE`, or `--on auto` or none for
    the machine `lane_machine` picks) waits for a slot of that machine's NAME pool instead of its merge-check slots (`pool_env`);
    `--timeout S` is the run's (`research run --timeout`, custody 24 h).
    `--shard auto:K/N --shard-plan-file PLAN` (with --on) is shard K of the auto plan PLAN, sent as the run's `AUTO_PLAN` input."""
    on = base = ratchet_base = pool = timeout = plan_file = None
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
        elif x == "--ratchet-base":
            ratchet_base = next(it)
        elif x == "--agreement":
            agreement = True
        elif x == "--priority":
            priority = True
        elif x == "--timeout":
            timeout = next(it)
        elif x == "--shard-plan-file":
            plan_file = Path(next(it)).resolve()
        else:
            rest.append(x)
    try:
        pooled = pool_env(pool, on, priority) + (["--timeout", timeout, "--custody-ttl", "24h"] if timeout else [])
        if pool is not None and on in (None, "auto"):
            on = lane_machine(pool)
    except ValueError as e:
        print(f"check: --pool {e}", file=sys.stderr)
        return 2
    combining = rest[:1] == ["--combine"]
    records = "--cc-records" in rest
    parts = None
    deferred = rest[rest.index("--deferred") + 1] if "--deferred" in rest[:-1] else None
    spec = rest[rest.index("--shard") + 1] if "--shard" in rest[:-1] and not records else None
    if (spec or "").startswith("auto:") != (plan_file is not None) or (plan_file is not None and not on):
        print(f"check: --shard {spec or 'K/N'}{' --shard-plan-file' * (plan_file is not None)} refused: an auto shard is "
              "`--shard auto:K/N --shard-plan-file PLAN --on M`, PLAN made once for all N (`--shard-plan --out PLAN`) and "
              "sent as the run's input, and only an auto shard takes a plan file", file=sys.stderr)
        return 2
    try:
        if records:                                    # circuit-check's records alone: its memory, no Lean
            parts = ["circuit-check"]
        elif plan_file is not None:                    # its plan is of those steps' jobs alone (`auto_shard`)
            jobs = DECLARATION.needed(JOBS, [s for s in deferred.split(",") if s]) if deferred is not None else None
            parts = shard(spec, gated=deferred is None, jobs=jobs, plan_file=plan_file)["parts"]
        elif spec is not None:
            parts = shard(spec, gated=deferred is None)["parts"]
            if deferred is not None:
                parts = [p for p in parts if p in deferred_parts(deferred)]
        elif deferred is not None:
            parts = deferred_parts(deferred)
    except DECLARATION.DeclarationError as e:
        print(f"check: {e}", file=sys.stderr)
        return 2
    min_mem = train_gb(priority, pool, parts, combining)
    pair, both = None, []
    if on == "auto" or (on and "," in on):
        names = list(AUTO) if on == "auto" else on.split(",")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        docs, got, why, skipped = place(names, head, priority, min_mem=min_mem)
        for s in skipped:
            print(f"check: --on {on}: skipping {s}", flush=True)
        if got is None:
            print(f"check: --on {on} refused: {why} ({', '.join(names)}); nothing launched", file=sys.stderr, flush=True)
            return 2
        print(f"check: --on {on}: {got}: {why}", flush=True)
        on = got
        if both := twins(names, docs, got, priority, min_mem):
            pair = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.urandom(2).hex()}"
            print(f"check: --on: twins {pair} on {', '.join(both)}: the first to take a check slot runs, and the others leave "
                  f"their line (exit {SLOT.NOT_STARTED}, not started)", flush=True)
    spec = importlib.util.spec_from_file_location("check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
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
    # an after-landing run is no check of the commit: recorded without the tool, so `research merge` never reads it.  On a machine
    # every run of a commit shares the shipped tree <root>/src/<sha>, and two checks' Lake builds there broke each other's .lake,
    # so each runs in its own clone of it
    cmd = [sys.executable, "-m", "research", "run", *(["--tool", "check"] if deferred is None and not records else []), "--project", "verity",
           "--cwd", "clone" if on else "source", *(x for p in DECLARED for x in ("--declared-output", p)), *pooled,
           *(["--env", f"{SLOT.MIN_MEM_ENV}={min_mem}"] if on and min_mem else [])]
    env = dict(os.environ)
    d = Path(tempfile.mkdtemp(prefix="check-lean-deps-"))
    if "lean-agreement" in need:
        try:
            signed = agreement_url_file(d)
        except NotSigned as e:
            shutil.rmtree(d, ignore_errors=True)
            print(f"check: --record refused: the run needs lean-agreement, which fetches the pinned upstream build and inputs "
                  f"through presigned URLs, and {e}; nothing launched", file=sys.stderr, flush=True)
            return 2
        pinned = json.loads(signed.read_text())
        print(f"check: changes under {', '.join(r for r in tool.CHECK.merge_requires['lean-agreement'] if not r.startswith('!'))} "
              f"against {base}: sending presigned URLs for the pinned upstream build and inputs "
              f"({sum(p['bytes'] for p in pinned.values()) >> 20} MiB, which the run fetches)")
        cmd += [x for k in ("upstream", "inputs") for x in ("--input", f"flock_{k}={pinned[k]['art']}")]
        if on:
            cmd += ["--send", str(signed)]
        else:
            env[AGREEMENT_URLS_ENV] = str(signed)
    audits = not combining and (parts is None or any(p.startswith("lean-audit:") for p in parts))
    if audits and (urls := lean_deps_url_file(d)):
        sent = json.loads(urls.read_text())
        shared = sum(k.startswith("lean-artifacts/") for k in sent)
        print(f"check: sending presigned URLs for {len(sent) - shared} pinned Lean dependency bundle(s) and {shared} Lean artifact "
              f"bundle(s) of recent checks, mark audits and records runs")
        if on:
            cmd += ["--send", str(urls)]
        else:
            env[LEAN_DEPS_URLS_ENV] = str(urls)
    if on and (combining or parts is not None):
        (d / SHARD_PARTS).write_text(json.dumps(parts or []))
        cmd += ["--send", str(d / SHARD_PARTS)]
    if plan_file is not None:
        shutil.copyfile(plan_file, d / AUTO_PLAN)
        cmd += ["--send", str(d / AUTO_PLAN)]
    if not combining and not records:
        if "--verdicts-in" in rest[:-1]:
            i = rest.index("--verdicts-in")
            rest[i + 1] = str(given := Path(rest[i + 1]).resolve())
            if not given.is_file():
                raise SystemExit(f"check: --verdicts-in {given}: no such pack")
            if why := VERDICTS.refusal(given):
                raise SystemExit(f"check: --verdicts-in {given}: refused ({why})")
        if urls := verdict_url_file(d, commit=_rev("HEAD") if "--no-cache" in rest else None):
            if on:
                cmd += ["--send", str(urls)]
            else:
                env[VERDICT_URLS_ENV] = str(urls)
    if on:
        cmd += ["--on", on, "--source", "."]
        if "--verdicts-in" in rest:                    # the pack travels with the run; on the machine it's found in the run's inputs/
            cmd += ["--send", str(Path(rest[rest.index("--verdicts-in") + 1]).resolve())]
        (d / BASE_FILE).write_text(f"{_rev(base) or base}\n")
        cmd += ["--send", str(d / BASE_FILE)]
        if ratchet_base is not None:
            (d / RATCHET_BASE_FILE).write_text(f"{_rev(ratchet_base) or ratchet_base}\n")
            cmd += ["--send", str(d / RATCHET_BASE_FILE)]
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


#: how many of the newest verdict packs of each kind of run (`PACKERS`) a recorded check imports (`verdict_packs`), beside
#: those of the marks of the PRs its tree holds
GATHER = 12
#: the bytes of recorded checks' packs one check imports (`verdict_packs`): a pack that would take them past this is passed over
#: by name, by the size the store lists, before anything fetches it, so what a run fetches and unpacks before its first step
#: stays bounded whatever a recorded check published.  The packs it passes over hold circuit-check's per-target stores, near
#: 1 GiB a pack, which no gated check reads (circuit-check is off the merge gate)
GATHER_BYTES = 256 << 20
#: how long the URLs `verdict_url_file` signs stay valid, in seconds: a run fetches them once it takes a slot, after its wait
VERDICT_URL_TTL_S = 6 * 3600


def verdict_url_file(d: Path, store=None, heads: list[str] | tuple[str, ...] | None = None,
                     commit: str | None = None) -> Path | None:
    """`VERDICT_URLS` in ``d`` (mode 600): the packs `verdict_packs` selects among those the store's remote holds, after the packs
    of ``commit``'s own recorded checks when it is given (`commit_packs`, which an uncached check compares with) and those of
    the runs `TRAIN_CARRY_FROM` names (`carried_packs`), each with a GET
    of its blob that the remote signs for `VERDICT_URL_TTL_S`, for the run to fetch and import before its steps
    (`verdicts_fetched`): the launcher lists and signs, and fetches nothing.  ``heads`` are `verdict_packs`'; the launcher, which
    gives neither them nor a ``store``, sends for the PRs this checkout's queue merges hold (`common.merged_heads`).  None, with
    the reason printed, when the store can't be read, no remote here signs URLs or no recorded check published a pack: the run
    then reuses only its node's own passes and a pack given."""
    from research.store.local import LocalStore, object_key

    try:
        if heads is None:
            heads = COMMON.merged_heads(ROOT) if store is None else ()
        store = store or LocalStore()
        if (sign := getattr(store.remote, "presign", None)) is None:
            print("check: verdicts: no store remote here signs URLs, so the run imports no recorded check's pack", flush=True)
            return None
        newest = verdict_packs(store, remote=True, heads=heads)
        own = commit_packs(store, commit, remote=True) if commit else []
        if own:
            print(f"check: verdicts: {len(own)} packs of {len({p['run'] for p in own})} recorded checks of {commit[:12]}, every "
                  f"pass it was recorded with: {', '.join(dict.fromkeys(p['run'] for p in own))}", flush=True)
        carried = carried_packs(store, remote=True)
        listed = list({p["sha256"]: p for p in [*own, *carried, *newest]}.values())
        packs = [{"run": p["run"], "sha256": p["sha256"], "bytes": p["bytes"], "url": sign(object_key(p["sha256"]), VERDICT_URL_TTL_S)}
                 for p in listed]
    except Exception as e:  # noqa: BLE001  (no URL: the check reuses only its node's passes and the one given)
        print(f"check: no verdict packs listed from the store ({type(e).__name__}: {e})", flush=True)
        return None
    if not packs:
        return None
    path = d / VERDICT_URLS
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(packs, f)
    runs = list(dict.fromkeys(p["run"] for p in packs))
    print(f"check: sending presigned URLs for the verdict packs of {len(runs)} recorded checks "
          f"({sum(p['bytes'] for p in packs) / 2**20:.0f} MiB): {', '.join(runs)}", flush=True)
    return path


#: the packs a recorded check writes in its run dir, which `record` declares outputs: the passes its steps stored (`Packs`), and
#: those it found before they ran (`Reused`)
REUSED_PACK = "verdicts-reused.tar.xz"
PACK_FILES = ("verdicts.tar.gz", REUSED_PACK)
#: what `record` declares outputs, which custody uploads however the run ends: the packs and the final live record
DECLARED = (*PACK_FILES, LIVE.LIVE_FILE)


def pack_file(store, rid: str, name: str = "verdicts.tar.gz") -> tuple[str, dict] | None:
    """(artifact, its listing's row: path, bytes, sha256, remote) of the pack ``name`` the attempt ``rid`` published among its run
    files, or in its run record when a SIGKILL left it no result.json to list them; None when it published none."""
    out = (store.get_attempt(rid) or {}).get("outputs") or {}
    for art in dict.fromkeys(a for a in (out.get("run_files"), out.get("run_record")) if a):
        for f in store.listing(art):
            if f["path"] == name:
                return art, f
    return None


def attempt_pack(store, rid: str, d: Path, name: str = "verdicts.tar.gz") -> Path | None:
    """The verdict pack the attempt ``rid`` published (`pack_file`), fetched under ``d``, or None."""
    if (got := pack_file(store, rid, name)) is None:
        return None
    return store.fetch(got[0], d / Path(name).name.split(".", 1)[0] / rid, paths=[name]) / name


#: the pack the Lean audit of a ready mark's merge leaves among its run's files (`tools/verity/lean/warm.py`'s `PACK`)
MARK_PACK = COMMON.MARK_AUDIT_OUT + "verdicts.tar.gz"
#: the runs `verdict_packs` lists packs of, each kind as (the store's attempts of it, its pack's path among its files): the
#: recorded checks, and the audits of ready marks' merges, whose packs hold Lean verdicts alone
PACKERS = ((COMMON.RECORDED_CHECKS, "verdicts.tar.gz"), (COMMON.MARK_AUDITS, MARK_PACK))


def verdict_packs(store, limit: int = GATHER, remote: bool = False, heads: list[str] | tuple[str, ...] = ()) -> list[dict]:
    """{run, sha256, bytes, path} of the pack of each of the ``limit`` runs of each kind of `PACKERS` that ended last and
    published one, the recorded checks' first and each kind's newest first: a failed or cancelled run's too, since its pack holds
    only the passes its steps stored, each under a key of everything it read.  Before the newest mark audits' come those of the
    marks of ``heads`` (`common.lookups`), however old and beside the ``limit``, as `lean_audit.shared_bundles` lists their
    bundles.  One whose listed size would take them past `GATHER_BYTES` is passed over by name, unfetched, and with ``remote`` so
    is one the store's remote doesn't hold, which no URL fetches.  When the store has a remote, the attempts started in the last
    `common.RECENT_S` are refreshed from it first."""
    if store.remote is not None:
        store.refresh(attempts_only=True, since=COMMON.recent(time.time()))
    ended = ("state IN ('done', 'failed', 'cancelled') "
             "ORDER BY julianday(start_utc) + COALESCE(wall_s, 0) / 86400.0 DESC LIMIT ?")
    packs, room = [], GATHER_BYTES
    for runs, name in PACKERS:
        for which, args, most in COMMON.lookups(runs, limit, heads):
            found = 0
            for (rid,) in store.index.sql(f"SELECT id FROM attempts WHERE {which} AND {ended}", args):
                if any(p["run"] == rid for p in packs):
                    continue
                if (got := pack_file(store, rid, name)) is None or (remote and not got[1].get("remote")):
                    continue
                if (size := got[1]["bytes"]) > room:
                    print(f"check: verdicts: skipping {rid}'s pack: its {size / 2**20:.0f} MiB would take this check's packs past "
                          f"{GATHER_BYTES / 2**20:.0f} MiB (GATHER_BYTES); its verdicts are recomputed", flush=True)
                    continue
                packs.append({"run": rid, "sha256": got[1]["sha256"], "bytes": size, "path": name})
                room -= size
                if (found := found + 1) == most:
                    break
    return packs


def commit_packs(store, commit: str, remote: bool = False) -> list[dict]:
    """{run, sha256, bytes} of both packs (`PACK_FILES`: the passes it stored and those it reused) of each recorded check of
    ``commit`` and each of its after-landing runs (`plan.own_checks`, newest first), the passes `check.py --plan` reads for it too:
    every pass the commit was recorded with, held to no `GATHER_BYTES`.  With ``remote``, one the store's remote doesn't hold is
    passed over, as no URL fetches it.  Read after `verdict_packs`, which refreshes the recent attempts."""
    return run_packs(store, _load("plan").own_checks(store, commit), remote)


def run_packs(store, rids: list[str], remote: bool = False) -> list[dict]:
    """{run, sha256, bytes} of both packs (`PACK_FILES`) of each run ``rids`` names that published them, in that order; with
    ``remote``, one the store's remote doesn't hold is passed over."""
    packs = []
    for rid in rids:
        for name in PACK_FILES:
            if (got := pack_file(store, rid, name)) is not None and (got[1].get("remote") or not remote):
                packs.append({"run": rid, "sha256": got[1]["sha256"], "bytes": got[1]["bytes"]})
    return packs


def carried_packs(store, remote: bool = False) -> list[dict]:
    """`run_packs` of the runs `TRAIN_CARRY_FROM` names: the verified queue's shards of the candidates it dropped that a rebuilt
    candidate replaces (`research.vqueue_slim.start`'s ``after``), whose passes its check reuses before any other's, so neither
    the newest checks' `GATHER` nor `GATHER_BYTES` leaves them out; each one named that published no pack is printed."""
    if not (rids := [r for r in (SETTINGS.TRAIN_CARRY_FROM or "").split(",") if r]):
        return []
    packs = run_packs(store, rids, remote)
    gone = [r for r in rids if not any(p["run"] == r for p in packs)]
    print(f"check: verdicts: {len(packs)} packs of the {len(rids) - len(gone)} runs TRAIN_CARRY_FROM names that published one, "
          f"held to no GATHER_BYTES: {', '.join(dict.fromkeys(p['run'] for p in packs)) or 'none'}"
          + (f"; none of {', '.join(gone)}" if gone else ""), flush=True)
    return packs


def gathered_packs(d: Path, store, limit: int = GATHER) -> list[tuple[str, Path]]:
    """(attempt, pack) of the newest packs `verdict_packs` selects (those of no PR's marks beside them), fetched under ``d``,
    newest first, for a plan read here (`plan.py`, `measure`); one import refuses is passed over by name."""
    packs = []
    for p in verdict_packs(store, limit):
        if (pack := attempt_pack(store, p["run"], d, p["path"])) is None:
            continue
        if why := VERDICTS.refusal(pack):
            print(f"check: verdicts: skipping {p['run']}'s pack, which import refuses ({why}); its verdicts are recomputed",
                  flush=True)
            continue
        packs.append((p["run"], pack))
    return packs


#: the pack paths of circuit-check's per-target cache (`groups`' `--cache`), which a run with `--no-cache --circuit-cache` reuses
CIRCUIT_ENTRIES = "check/circuit-check/"
#: the pack paths of the suites' and tests' passes (`$VERITY_TEST_CACHE`), what a run with `--no-cache` imports of a pack (and
#: `CIRCUIT_ENTRIES` with `--circuit-cache`): it reuses none of them, and its fresh suites name each failure one of them answers
#: for (`suites.py --fresh`'s `contradicts`)
COMPARED = "tests/"


def _only(only: str | tuple[str, ...] | None) -> str:
    return f" (only {', '.join((only,) if isinstance(only, str) else only)})" if only else ""


def verdicts_in(pack: Path, run_dir: Path | None, only: str | tuple[str, ...] | None = None,
                named: dict[str, int] | None = None) -> dict:
    """Import a verdict pack before the steps run (`verdicts.py`): the path given, else its name in a recorded run's `inputs/`;
    with ``only``, just its entries under that pack path (or those).  ``named`` takes the circuit records it names
    (`records.named_in`)."""
    try:
        here = pack.exists()
    except PermissionError:                  # the launcher's path, under a directory this machine can't read (its /root)
        here = False
    if not here and run_dir is not None and (run_dir / "inputs" / pack.name).exists():
        pack = run_dir / "inputs" / pack.name
    added, kept = VERDICTS.import_(pack, only)
    if named is not None:
        named.update(RECORDS.named_in(pack))
    print(f"check: verdicts: {added} imported, {kept} already here, from {pack.name}{_only(only)}", flush=True)
    return {"pack": pack.name, "sha256": _sha256(pack), "added": added, "already": kept, **({"only": only} if only else {})}


#: how long a fetch through a presigned URL may wait on the store's answer before it gives up, in seconds
FETCH_TIMEOUT_S = 120


def fetch_signed(p: dict, dest: Path) -> str | None:
    """Fetch the file ``p`` lists (a recorded check's pack, `take_verdict_urls`, or a pinned file of the agreement's,
    `agreement_sent`) through its presigned URL to ``dest``, never more bytes than it lists; None once the bytes are the ones
    listed (their size and sha256), else why not, which never holds the URL."""
    h, n = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(p["url"], timeout=FETCH_TIMEOUT_S) as r, open(dest, "wb") as f:  # noqa: S310  (a presigned URL)
            while n <= p["bytes"] and (chunk := r.read(1 << 20)):
                h.update(chunk)
                f.write(chunk)
                n += len(chunk)
    except (OSError, ValueError, http.client.HTTPException) as e:     # urllib's errors, which name the URL, aren't printed
        return f"it can't be fetched ({type(e).__name__}{f' {e.code}' if isinstance(e, urllib.error.HTTPError) else ''})"
    if (n, h.hexdigest()) != (p["bytes"], p["sha256"]):
        return f"the {n} bytes fetched aren't the {p['bytes']} the store lists under its sha256"
    return None


def verdicts_fetched(listed: list[dict], only: str | tuple[str, ...] | None = None,
                     named: dict[str, int] | None = None) -> list[dict]:
    """Fetch each pack ``listed`` (`take_verdict_urls`, newest first) and import it before the steps run (`verdicts.import_`, which
    never replaces an entry here, so where two packs hold one entry the newer's is kept); with ``only``, just its entries under
    that pack path (or those).  A pack that can't be fetched, isn't the one listed or that import refuses is passed over by
    name, so its verdicts are recomputed.  ``named`` takes the circuit records each pack imported names (`records.named_in`).
    Each pack's record for the result's `verdicts`."""
    got, t0 = [], time.monotonic()
    with tempfile.TemporaryDirectory(prefix="check-verdicts-") as d:
        pack = Path(d) / "verdicts.tar.gz"
        for p in listed:
            if (why := fetch_signed(p, pack)) is None:
                try:
                    added, kept = VERDICTS.import_(pack, only)
                except VERDICTS.UNREADABLE as e:
                    why = f"import refuses it ({type(e).__name__}: {e})"
                else:
                    if named is not None:
                        named.update(RECORDS.named_in(pack))
                    got.append({"run": p["run"], "sha256": p["sha256"], "added": added, "already": kept,
                                **({"only": only} if only else {})})
                    continue
            print(f"check: verdicts: skipping {p['run']}'s pack: {why}; its verdicts are recomputed", flush=True)
            got.append({"run": p["run"], "sha256": p["sha256"], "skipped": why})
    taken = [g for g in got if "skipped" not in g]
    print(f"check: verdicts: {sum(g['added'] for g in taken)} imported, {sum(g['already'] for g in taken)} already here, from "
          f"{len(taken)} of {len(listed)} recorded checks' packs{_only(only)} in {time.monotonic() - t0:.0f} s", flush=True)
    return got


#: the suite whose session builds circuit-check's binding records apart, as its step does (`records.py`)
CIRCUIT_SUITE = "tools/verity/circuit_check"


def builds_records(only: list[str] | None) -> bool:
    """Whether a check of the parts ``only`` (all of it when None) runs circuit-check's step or `CIRCUIT_SUITE` (its part, or the
    whole of `pytest`), the two that read and write its binding records."""
    if only is None:
        return True
    names = {"", *(s.name for s in SUITES.discover() if s.path == CIRCUIT_SUITE)}
    return any(p.partition(":")[0] == "circuit-check" or (p.partition(":")[0] == "pytest" and SHARDS.suite_part(p)[1] in names)
               for p in only)


def cc_manifests() -> bool:
    """Whether this tree's circuit-check writes the manifest of the records its catalog took or built (`records.RUN_MANIFESTS`:
    the step's `--records-manifest`, a session's `records.SESSION_ENV`); a tree whose circuit-check predates it is checked too."""
    cli = ROOT / CIRCUIT_SUITE / "src" / "circuit_check" / "cli.py"
    return cli.is_file() and "--records-manifest" in cli.read_text()


def store_remote():
    """The store's remote through the run's custody key (`custody_dir`), else None, with the reason printed."""
    if (custody := custody_dir()) is None:
        print("check: circuit records: no custody key here, so none is fetched or published", flush=True)
        return None
    try:
        return LIVE.custody_remote(custody)
    except Exception as e:  # noqa: BLE001  (no remote: circuit-check builds what the packs held no record of)
        print(f"check: circuit records: no store remote ({type(e).__name__}); none is fetched or published", flush=True)
        return None


def records_fetched(named: dict[str, int], remote) -> dict:
    """Fetch the circuit records ``named`` that circuit-check's cache lacks from ``remote`` (`records.fetch`), before the steps:
    the counts for the result's `verdicts`, printed.  Nothing here fails the check: what isn't fetched is built."""
    try:
        got = RECORDS.fetch(named, cache_dir() / "circuit-check", RECORDS.getter(remote), base=cache_dir())
    except Exception as e:  # noqa: BLE001  (a fetch that breaks costs the builds, never a verdict)
        print(f"check: circuit records: fetching failed ({type(e).__name__}: {e}); circuit-check builds them", flush=True)
        return {"named": len(named), "error": type(e).__name__}
    print(RECORDS.fetch_line(got), flush=True)
    return got


#: The suites whose tests run `lake` or what the Lean steps build under `.lake/`: a step of the Lean group once `lean-build` has
#: built the verifier, with elan's `lake` on the PATH, so what they run or skip never depends on how far a concurrent
#: `lake build` has got, and no test's `lake build` shares a build directory with one: lean-audit, which runs
#: beside them, builds only in trees of its own (tests/test_no_wall_clock.py holds every other suite to not skipping on them).
#: `suites.py` run on its own builds the verifier before them itself; here it is told the build is done (`--lean-built`).
AFTER_LEAN = SUITES.AFTER_LEAN

#: The opt-in tests a cold check runs by name (`--no-cache`, unsharded and without `--deferred`: the nightly `audit-main`), each
#: suite's in a `suites.py` run of its own with its variables set, in the step that runs the suite, after the suite's own run
#: (`opt_in`): suite -> each variable its opt-in tests read -> those tests by node id in the suite.  Each skips in every other
#: check by a mark in its own file that reads its variable (tests/test_opt_in_tests.py), so no candidate's check pays for it and
#: no suite's key holds a variable; a test a variable turns on that this doesn't name, circuit-check's serial-report test of
#: about 2.5 h, stays off.
OPT_IN = {
    "tools/verity/circuit_check": {"VERITY_CIRCUIT_CHECK_SLOW": (
        "tests/test_circuit_check.py::test_every_call_the_suite_checks_is_partitioned",
        "tests/test_circuit_check.py::test_a_call_whose_partition_is_unchecked_passes_call_status_to_its_callees")},
    "verity/definitions": {"VERITY_ML_DEFINITIONS_SLOW": (
        "tests/test_dsa_topk_reads.py::test_the_served_merge_with_ports_a_block_is_the_one_port_merge_gate_for_gate",)},
    "verity/protocols/remote/verifier/composition": {"VERITY_SLOW": (
        "tests/test_composition_difftest.py::test_remote_agrees",
        "tests/test_composition_difftest.py::test_onsite_agrees",
        "tests/test_composition_difftest.py::test_product_over_remotes_subprotocols_is_remote",
        "tests/test_composition_difftest.py::test_a_reference_without_reject_dominance_is_caught")},
    "adapters/vllm": {
        "VERITY_VLLM_INTERIOR_TABLE": (
            "tests/pipeline/test_interior_table.py::test_an_interior_opens_the_tables_round_1_served_and_the_challenger_resolves_its_reads_to_them",
            "tests/pipeline/test_interior_table.py::test_round_2_derives_no_links_from_a_block_that_returns_a_read_of_two_outputs",
            "tests/pipeline/test_interior_table.py::test_round_2_reads_root_and_domain_at_the_name_the_served_records_ports_give_their_links",
            "tests/pipeline/test_interior_table.py::test_a_table_round_1_did_not_register_is_refused",
            "tests/pipeline/test_interior_table.py::test_every_table_of_a_class_publishes_its_c_strings_whatever_its_length",
            "tests/pipeline/test_interior_table.py::test_a_table_reads_interior_without_a_test_key_registers_the_same_root_and_domain",
            "tests/pipeline/test_interior_table.py::test_a_read_that_does_not_open_its_table_is_refused_before_anything_is_committed",
            "tests/pipeline/test_interior_table.py::test_a_table_read_interior_is_refused_without_its_tables_or_with_tables_it_cannot_read",
            "tests/pipeline/test_interior_table.py::test_the_challengers_history_refuses_a_table_round_1_did_not_register"),
        "VERITY_VLLM_HISTORY_E2E": (
            "tests/pipeline/test_serving_rows_history.py::test_the_window_commits_the_writers_rows_as_its_tables_strings_and_the_verifier_pins_each_root",)},
}
#: where a cold check's opt-in runs write each suite's outcome and log, apart from its steps' own (`suites.json`, `suites/`)
OPT_IN_DIR = "opt-in"


def then_opt_in(argv: list[str], out: Path, step: str, args: list[str]) -> str:
    """``argv`` and then, whether or not it passed, the opt-in tests of ``step`` (`opt_in`, with suites.py's ``args``), as one
    shell command that fails when either fails."""
    them = [sys.executable, str(Path(__file__).resolve()), "--opt-in", str(out), step, *args]
    return "rc=0; " + "; ".join(f"{shlex.join(c)} || rc=$?" for c in (argv, them)) + "; exit $rc"


def opt_in(out: Path, step: str, args: list[str]) -> int:
    """`check.py --opt-in OUT STEP ARGS...`, a cold check's opt-in tests (`OPT_IN`) of the suites step STEP runs (pytest: those not
    in `AFTER_LEAN`; lean-suites: those in it; all: every one, on a tree without the verifier), each suite's in a `suites.py` run
    of its own with ARGS, its variables set to 1 and its tests named, with its outcome and log in OUT's `OPT_IN_DIR`.  0 only when
    each named test passed: one that skipped, or that its suite doesn't have, fails the step."""
    if step not in ("pytest", "lean-suites", "all"):
        print(f"check: --opt-in OUT STEP: STEP is pytest, lean-suites or all, not {step!r}", file=sys.stderr)
        return 2
    names = {s.path: s.name for s in SUITES.select(SUITES.discover(), list(OPT_IN))}
    d = out / OPT_IN_DIR
    d.mkdir(parents=True, exist_ok=True)
    rc = 0
    for path, by_var in OPT_IN.items():
        if step != "all" and (names[path] in AFTER_LEAN) != (step == "lean-suites"):
            continue
        tests, on = [t for of in by_var.values() for t in of], ", ".join(f"{var}=1" for var in by_var)
        j = d / f"{path.replace('/', '_')}.json"
        j.unlink(missing_ok=True)
        print(f"check: opt-in: {path} with {on}: {', '.join(t.partition('::')[2] for t in tests)}", flush=True)
        ran = subprocess.run([sys.executable, str(ROOT / "tools" / "verity" / "check" / "suites.py"), path, "--json", str(j),
                              "--logs", str(d), *args, "--", *tests], cwd=ROOT,
                             env={**os.environ, **dict.fromkeys(by_var, "1")}).returncode
        c = (next(iter(json.loads(j.read_text())), {}).get("counts") or {}) if j.exists() else {}
        if ran or c.get("passed", 0) < len(tests) or any(c.get(k) for k in ("failed", "error", "skipped")):
            why = ", ".join(f"{v} {k}" for k, v in c.items() if k != "passed" and v) or f"suites.py exited {ran}"
            print(f"check: opt-in: {path}: {c.get('passed', 0)} of the {len(tests)} tests named with {on} passed ({why}); "
                  f"its log is in {d}", flush=True)
            rc = 1
    return rc


#: what pytest and the Lean steps hold beside circuit-check in one run, in GiB: pytest and lean-audit have held 160 to 350
#: between them, and circuit-check, given the slot's whole mem=, took the run's scope to its cap (an OOM kill)
SIBLINGS_GB = 350


def cc_mem_gb(slot_gb: int, beside: bool, lean: bool = True) -> int:
    """circuit-check's --mem-gb in a slot of ``slot_gb``: all of it alone (a shard without pytest or Lean), else what
    `SIBLINGS_GB` leaves, but never under a quarter of the slot; beside pytest without ``lean`` (an after-landing run of
    circuit-check and pytest), what pytest's share of that (`tests_gb`, half, the other half lean-suites') leaves."""
    if not beside:
        return slot_gb
    shared = max(slot_gb - SIBLINGS_GB, slot_gb // 4)
    return shared if lean else slot_gb - (slot_gb - shared) // 2


#: where circuit-check finds the memory it is granted once it runs alone in the slot (its ``--mem-gb-file``, `grant_when_alone`)
CC_GRANT = "circuit-check.mem-gb"
#: where a check that holds some of circuit-check's parts writes each one's records and manifest, ``K`` of ``K/N`` (its
#: ``--shard-dir``), published with the result for `circuit_join` to read
CC_SHARDS = "circuit-check-shards"


def circuit_shards(only: list[str] | None) -> list[str] | None:
    """The shards of circuit-check's targets (``K/N``) a check of the parts ``only`` runs, or None when it runs `--all` whole
    (unsharded, or holding all of circuit-check's parts) or not at all."""
    if only is None:
        return None
    held = [p.partition(":")[2] for p in only if p.partition(":")[0] == "circuit-check" and p.partition(":")[2]]
    every = DECLARATION.job_parts(next(j for j in JOBS if j.name == "circuit-check"), {}) or []
    return held if held and sorted(held) != sorted(every) else None


#: the records step (`--cc-records`): every catalog binding's record, built once (`circuit-check --all --records-only`) and tarred
#: under the run's $OUT with the tree it was built from (`CC_RECORDS_TREE`), its sha256 in the result (`CC_RECORDS_KIND`'s
#: ``records``); drive's sharded after-landing run builds it before the shards and gives each its artifact
#: (`--cc-records-from ART@SHA256`), so no shard's circuit-check builds the bindings itself
CC_RECORDS = "out/circuit-records.tar.gz"
CC_RECORDS_KIND = "circuit-records/v1"
CC_RECORDS_TREE = "tree"


def cc_records(run_dir: Path, n: int | None = None) -> int:
    """`--cc-records`, inside a recorded run: `circuit-check --all --records-only` into a fresh cache, its report
    `circuit-records.json`, then the cache and this tree's id tarred to `CC_RECORDS`, and ``result.json`` (`CC_RECORDS_KIND`)
    naming the tarball's path, sha256 and tree; exit circuit-check's, and no tarball unless it passed."""
    tree, t0 = _rev("HEAD^{tree}"), time.monotonic()
    d = Path(tempfile.mkdtemp(prefix="check-cc-records-"))
    try:
        gb = slot_mem_gb()
        cmd = [sys.executable, "-m", "circuit_check", "--all", "--records-only", "-q", "--cache", str(d / "cache"),
               "--jobs", str(n or cores()), *(["--mem-gb", str(gb)] if gb is not None else []),
               "--out", str(run_dir / "circuit-records.json")]
        rc = subprocess.run(cmd, cwd=ROOT, env=env()).returncode
        records = None
        if rc == 0 and tree:
            (d / CC_RECORDS_TREE).write_text(tree + "\n")
            (tar := run_dir / CC_RECORDS).parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar, "w:gz") as t:
                t.add(d / CC_RECORDS_TREE, CC_RECORDS_TREE)
                t.add(d / "cache", "cache")
            records = {"path": CC_RECORDS, "sha256": _sha256(tar), "tree": tree}
    finally:
        shutil.rmtree(d, ignore_errors=True)
    secs = round(time.monotonic() - t0, 1)
    doc = {"schema": "research/result/v0.1", "kind": CC_RECORDS_KIND, "run_id": SETTINGS.require("RESEARCH_RUN_ID"),
           "workload_fingerprint": {"tree": tree},
           "validation": {"status": "passed" if records else "failed", "checker": "tools/verity/check/check.py --cc-records",
                          "detail": f"circuit-check --all --records-only exited {rc}" + ("" if tree else "; no tree")},
           "measurements": [{"name": "circuit-records_wall", "value": secs, "unit": "s"}],
           "artifacts": [p for p in (CC_RECORDS, "circuit-records.json") if (run_dir / p).exists()],
           **({"records": records} if records else {})}
    (run_dir / "result.json.tmp").write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(run_dir / "result.json.tmp", run_dir / "result.json")
    print(f"check: --cc-records: {'records ' + records['sha256'][:12] + ' in ' + CC_RECORDS if records else 'FAILED'} in {secs} s",
          flush=True)
    return 0 if records else rc or 1


def fetch_records(art: str, d: Path) -> Path:
    """The records tarball (`CC_RECORDS`) of the run record ``art``, fetched under ``d`` through the run's custody key (the
    module docstring's "The evidence store") or, outside a recorded run, this machine's store."""
    from research.store.custody import custody_env
    from research.store.local import LocalStore

    with custody_env(c) if (c := custody_dir()) else contextlib.nullcontext():
        return Path(LocalStore().fetch(art, d, paths=[CC_RECORDS])) / CC_RECORDS


def cc_records_in(spec: str, cache: Path, fetch=fetch_records) -> str | None:
    """`--cc-records-from ART@SHA256`: the records step's tarball (`cc_records`), from the run record ART, unpacked into
    circuit-check's cache ``cache`` once its bytes have that sha256 and the tree it was built from is this one; else why it is
    refused, and nothing unpacked.  Each record goes in by rename, so a circuit-check reading the cache meanwhile sees it whole
    or not at all, and one counts only while its own reads still match (`circuit_check.units`).  The tarball is fetched and
    unpacked in a hidden directory inside the cache as its path resolves, so no rename crosses a filesystem when that path is a
    link or mount onto another disk; circuit-check lists records only in their key's directory and `verdicts` walks no hidden
    directory, so neither reads it."""
    art, _, sha = spec.partition("@")
    if not art.startswith("art:") or not re.fullmatch(r"[0-9a-f]{64}", sha):
        return f"{spec} is not ART@SHA256"
    cache.mkdir(parents=True, exist_ok=True)
    root = cache.resolve()
    d = Path(tempfile.mkdtemp(prefix=".cc-records-", dir=root))
    try:
        try:
            tar = fetch(art, d)
        except Exception as e:  # noqa: BLE001  (refused, and the reason named)
            return f"{art} can't be fetched ({type(e).__name__}: {e})"
        if (got := _sha256(tar)) != sha:
            return f"its bytes' sha256 is {got}, not {sha}"
        with tarfile.open(tar) as t:
            try:
                built = t.extractfile(CC_RECORDS_TREE).read().decode().strip()
            except (KeyError, AttributeError):
                return f"it holds no {CC_RECORDS_TREE}, the tree it was built from"
            if built != (tree := _rev("HEAD^{tree}")):
                return f"it was built from tree {built[:12]}, and this tree is {(tree or 'unknown')[:12]}"
            t.extractall(d / "x", filter="data")
        n = 0
        for p in sorted((d / "x" / "cache").rglob("*.json")):
            (to := root / p.relative_to(d / "x" / "cache")).parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, to)
            n += 1
        print(f"check: circuit-check: {n} binding record(s) from {art} ({sha[:12]}) into {cache}", flush=True)
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def slot_mem_gb() -> int | None:
    return int(m) if (m := SETTINGS.get(SLOT.SLOT_MEM_ENV)).isdigit() else None


def grant_when_alone(path: Path, beside: list[threading.Event], gb: int) -> None:
    """Write ``gb`` (the slot's mem=) to ``path`` once every step beside circuit-check has ended: from then on it runs alone in
    the slot, and the targets it has not started may take all of it (b103: pytest ended 83 minutes into its 125)."""
    for e in beside:
        e.wait()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{gb}\n")
    os.replace(tmp, path)


def ratchet_gate(out: Path, cache: bool) -> list[str] | str:
    """ratchet-gate's command: `ratchet_gate.py BASE HEAD` as BASE has it, BASE being the commit in the run's `RATCHET_BASE_FILE`,
    else its `BASE_FILE` (`--record --on` sends them), else origin/main, with its verdicts kept under the check's cache unless
    ``cache`` is off.  A BASE at or above HEAD would judge nothing, so the step fails (`_step`: a reason that isn't a skip)."""
    sent = next((f for f in (out / "inputs" / RATCHET_BASE_FILE, out / "inputs" / BASE_FILE) if f.is_file()), None)
    named = sent.read_text().strip() if sent else "origin/main"
    if not (base := _rev(f"{named}^{{commit}}")):
        return f"skipped: no base: {named} names no commit here"
    if subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", base], cwd=ROOT, capture_output=True).returncode == 0:
        return (f"failed: its base {named} is HEAD or above it, so no change lies above the base to judge: check this commit "
                "against one below it (`--record --base REF` or `--ratchet-base REF`; drive's after-landing run sends one below "
                "its tip)")
    keep = str(cache_dir() / "ratchet-gate") if cache else ""
    script = ('g=$(mktemp -d) && trap \'rm -rf "$g"\' EXIT && git show "$1:$2" > "$g/ratchet_gate.py" && '
              'RATCHET_GATE_CACHE="$3" "$4" "$g/ratchet_gate.py" "$1" HEAD')
    return ["bash", "-c", script, "ratchet-gate", base, RATCHET_GATE, keep, sys.executable]


def suite_args(held: list[str], splits: dict[str, dict] | None, out: Path) -> list[str]:
    """`suites.py`'s arguments for a shard's suites of one step (``held``: `SUITE` or `SUITE:K/M`): each suite once, a split one
    as `SUITE:K1,K2/M` with the parts it holds, pinned to the partition ``splits`` names (`@` its digest) when it names one; and
    `--timings` for those whose timing record ``splits`` carries (an auto plan's), written under ``out``, so each shard of the
    plan partitions as the planner did."""
    ks: dict[str, list[int]] = {}
    ms: dict[str, int] = {}
    for x in held:
        name, _, spec = x.rpartition(":")
        if name and re.fullmatch(r"[1-9]\d*/[1-9]\d*", spec):
            ks.setdefault(name, []).append(int(spec.partition("/")[0]))
            ms[name] = int(spec.partition("/")[2])
        else:
            ks.setdefault(x, [])
    splits = splits or {}
    args = []
    for name, k in ks.items():
        pin = f"@{splits[name]['digest'][:12]}" if name in splits and splits[name].get("m") == ms.get(name) else ""
        args.append(f"{name}:{','.join(map(str, sorted(k)))}/{ms[name]}{pin}" if k else name)
    timings = {name: v["timings"] for name, v in splits.items() if name in ms and isinstance(v.get("timings"), dict)}
    if timings:
        f = out / f"suite-timings-{hashlib.sha256(json.dumps(timings, sort_keys=True).encode()).hexdigest()[:12]}.json"
        out.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(timings, sort_keys=True) + "\n")
        args += ["--timings", str(f)]
    return args


def groups(out: Path, n: int, cache: bool = True, urls: Path | None = None,
           only: list[str] | None = None, regenerate: bool = True,
           circuit_cache: bool | None = None, splits: dict[str, dict] | None = None,
           bookkeeping: bool = False, agreement_urls: Path | None = None) -> list[list[tuple[str, list[str] | str, Path]]]:
    """The independent groups, each a sequence of (name, argv or the reason it is skipped / cannot run, cwd); with ``only``, a
    shard's parts (`shard`), just the jobs those name, lean-audit on its packages there, and all the cores to pytest or to
    circuit-check when the other isn't among them.  ``circuit_cache`` (default ``cache``) is whether circuit-check reuses its
    per-target passes.  A split suite's parts (`pytest:SUITE:K/M`) run as `suites.py SUITE:K/M` (`suite_args`, with ``splits``,
    an auto plan's or the task list's partitions).  ``urls`` and ``agreement_urls`` are the files of presigned URLs the run was
    sent (`take_lean_deps_urls`, `take_agreement_urls`)."""
    circuit_cache = cache if circuit_cache is None else circuit_cache
    py = sys.executable
    a = max(1, n // 2)
    b = max(1, n - a)
    if only is not None:
        # and the builds they need, beside its parts when another shard holds them (an auto plan's, `shards.py`; a hand-written
        # plan's lean-build, beside some of lean-suites' parts)
        names = DECLARATION.needed(JOBS, sorted({p.partition(":")[0] for p in only}))
        a, b = (n if not names & {"flock-circuit-build", "circuit-check", "flock-rust-tests"} else a), (n if "pytest" not in names else b)
    with_lean = only is None or any(x.startswith("lean") for x in names)
    beside = with_lean or "pytest" in names
    suites = [py, str(ROOT / "tools" / "verity" / "check" / "suites.py")]
    slot_gb = slot_mem_gb()
    with_cc = only is None or "circuit-check" in names
    # pytest and lean-suites run at once, each with this share: what circuit-check beside them both leaves, halved; beside
    # circuit-check alone the other half is circuit-check's (`cc_mem_gb`), and alone in the slot pytest has all of it
    tests_gb = None if slot_gb is None else (slot_gb - cc_mem_gb(slot_gb, beside) if with_cc else slot_gb) // (
        2 if with_lean or with_cc else 1)
    tail = ["--timeouts", "--jobs", str(a), *([] if cache else ["--fresh"]), *(["--mem-gb", str(tests_gb)] if tests_gb is not None else []),
            *(["--run", rid] if (rid := SETTINGS.RESEARCH_RUN_ID) else [])]
    lean = lean_steps(out, cache, urls, regenerate, bookkeeping, agreement_urls)
    # an auto shard's suites (`pytest:SUITE`, `lean-suites:SUITE`): only those
    held = {job: [p.partition(":")[2] for p in only or () if p.partition(":")[0] == job and p.partition(":")[2]]
            for job in SHARDS.SUITE_JOBS}
    after_lean: list[str] = []
    # a cold check, unsharded and without --deferred, also runs the opt-in tests, by name with their variables set (`OPT_IN`)
    cold = not cache and only is None
    if len(lean) > 1:
        after_lean = [x for s in AFTER_LEAN for x in ("--skip", s)]
        lean_suites = [*suites, *(suite_args(held["lean-suites"], splits, out) or AFTER_LEAN), "--lean-built",
                       "--json", str(out / "suites-lean.json"),
                       "--logs", str(out / "suites"), *tail]
        lean.insert(3, ("lean-suites", ["bash", "-c", ELAN + (then_opt_in(lean_suites, out, "lean-suites", ["--lean-built", *tail])
                                                              if cold else shlex.join(lean_suites))], ROOT))
    pytest = [*suites, *suite_args(held["pytest"], splits, out), "--json", str(out / "suites.json"), "--logs", str(out / "suites"),
              *after_lean, *tail]
    if cold:
        pytest = ["bash", "-c", then_opt_in(pytest, out, "pytest" if len(lean) > 1 else "all", tail)]
    manifests = circuit_cache and cc_manifests()

    def cc_argv(tag: str) -> list[str]:
        return [py, "-m", "circuit_check", "--all", "-q", "--jobs", str(b), "--out", str(out / "circuit-check.json"),
                *(["--cache", str(cache_dir() / "circuit-check"), "--catalog-units"] if circuit_cache else []),
                *(["--records-manifest", str(out / f"circuit-records.{tag}.json")] if manifests else []),
                *(["--mem-gb", str(cc_mem_gb(slot_gb, beside, with_lean))] if slot_gb is not None else []),
                *(["--mem-gb-file", str(out / CC_GRANT)] if slot_gb is not None and beside else [])]
    cc = cc_argv("step")
    if held := circuit_shards(only):
        each = [[*cc_argv(f"step-{s.partition('/')[0]}"), "--shard", s, "--shard-dir", str(out / CC_SHARDS / s.partition("/")[0])]
                for s in held]
        cc = each[0] if len(each) == 1 else ["bash", "-c", "rc=0; " + "; ".join(f"{shlex.join(c)} || rc=$?" for c in each) + "; exit $rc"]
    build = ["bash", str(ROOT / "verity" / "protocols" / "flock" / "prover" / "check_build.sh")]
    rust = ["env", f"CARGO_BUILD_JOBS={b}", f"RUST_TEST_THREADS={b}", *build, "test"]
    built = {"pytest": ("pytest", pytest, ROOT), "flock-circuit-build": ("flock-circuit-build", build, ROOT),
             "circuit-check": ("circuit-check", cc, ROOT), "flock-rust-tests": ("flock-rust-tests", rust, ROOT),
             "ratchet-gate": ("ratchet-gate", ratchet_gate(out, cache), ROOT)}
    if len(lean) > 1:
        built |= {s[0]: s for s in lean}
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
#: an after-landing run's result kind (`--deferred`), which no gate takes: drive reads its steps and suites to name a red's culprit
DEFERRED_KIND = "check-deferred/v1"


def deferred_parts(steps: str) -> list[str]:
    """`--deferred STEP,...`: every part an unsharded check covers that is one of those ci.toml jobs or a job they need or run
    after (`declaration.needed`).  DeclarationError for a name no job has."""
    chosen = DECLARATION.needed(JOBS, [s for s in steps.split(",") if s])
    return [p for p in shard_parts() if p.partition(":")[0] in chosen]


def shard(spec: str, gated: bool = True, jobs: set[str] | None = None, plan_file: Path | None = None) -> dict:
    """`--shard K/N`: {"k", "n", "parts": shard K's parts under ci.toml's plan with N shards, but those of a job off the merge
    gate unless ``gated`` is False (an after-landing `--deferred` shard runs them, and its last shard also the parts no shard
    of the plan holds, ratchet-gate's where the plan doesn't name it, so the N shards together run every part drive's
    after-landing run passes), "all": every part a recorded unsharded check covers (`gated_parts`)}, on this tree's Lake
    packages, each spec's audit on its provers' shard (`declaration.resolve`, `audit_provers`).  DeclarationError when there is
    no such plan or shard, or the tree can't be listed.  `auto:K/N`: shard K of the auto plan (`auto_shard`)."""
    if spec.startswith("auto:"):
        return auto_shard(spec, gated, plan_file, jobs)
    m = re.fullmatch(r"(\d+)/(\d+)", spec)
    plans = DECLARATION.plans()
    if not m or int(m.group(2)) not in plans or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise DECLARATION.DeclarationError(f"--shard {spec}: ci.toml's plans are {', '.join(f'K/{n}' for n in sorted(plans))}")
    k, n = int(m.group(1)), int(m.group(2))
    parts_of = _parts_of()
    shards = DECLARATION.resolve(plans[n], JOBS, parts_of, audit_provers(parts_of["lean-audit"]))
    held = {p for s in shards for p in s}
    mine = set(shards[k - 1]) | ({p for p in DECLARATION.all_parts(JOBS, parts_of) if p not in held} if k == n and not gated else set())
    return {"k": k, "n": n,
            "parts": [p for p in DECLARATION.all_parts(JOBS, parts_of)
                      if p in mine and (not gated or p.partition(":")[0] not in OFF_GATE)],
            "all": DECLARATION.all_parts(DECLARATION.gated(JOBS), parts_of)}


#: `--shard auto:K/N`, an auto plan's shard
AUTO_SHARD = re.compile(r"auto:([1-9]\d*)/([1-9]\d*)")


def auto_parts(gated: bool = True, jobs: set[str] | None = None, circuit_shards: int | None = None,
               splits: dict[str, int] | None = None) -> list[str]:
    """The auto plan's parts on this tree (`shards.parts`): those of the jobs on the gate (or, not ``gated``, of every job),
    of ``jobs`` alone when given; circuit-check in ``circuit_shards`` parts when given, else its ci.toml `shards`; each suite
    ``splits`` names in its M parts, and each that declares parts in those (`suite_parts`)."""
    js = [replace(j, shards=circuit_shards) if circuit_shards and j.name == "circuit-check" else j
          for j in (DECLARATION.gated(JOBS) if gated else JOBS) if jobs is None or j.name in jobs]
    return SHARDS.parts(DECLARATION.all_parts(js, _parts_of()), [s.name for s in SUITES.discover()], list(AFTER_LEAN),
                        (FLOCK_LEAN / "lakefile.toml").exists(), splits, suite_parts())


def suite_splits(suite_s: dict[str, float], times: dict[str, dict[str, float]], split_s: float = SHARDS.SPLIT_S) -> dict[str, dict]:
    """The suites of this tree measured past ``split_s`` (``suite_s``, by name) split by test file (`shards.split_m`), but those
    that declare parts by test id: {name: {"m", "seconds", "digest": its partition's (`suites.partition` by ``times``, its files'
    recorded seconds), "timings": those of its files, "shares": each part's share of its seconds, "files": each part's test files}}."""
    out: dict[str, dict] = {}
    if not any(v is not None and v > split_s for v in suite_s.values()):
        return out
    tree, pkgs = SUITES.worktree(), SUITES.members()
    for s in SUITES.discover():
        if s.parts or suite_s.get(s.name) is None or suite_s[s.name] <= split_s:
            continue
        files = SUITES.test_files(s, tree, SUITES.dropped(s, SUITES.inputs_of(s, pkgs)))
        m = SHARDS.split_m(suite_s[s.name], len(files), split_s)
        if m < 2:
            continue
        mine = {f: v for f, v in (times.get(s.name) or {}).items() if f in set(files)}
        parts = SUITES.partition(files, mine, m)
        cost = SUITES.file_costs(files, mine)
        total = sum(cost.values()) or 1.0
        out[s.name] = {"m": m, "seconds": round(suite_s[s.name], 1), "digest": SUITES.partition_digest(parts), "timings": mine,
                       "shares": [round(sum(cost[f] for f in p) / total, 6) for p in parts], "files": parts}
    return out


def split_shares(splits: dict[str, dict]) -> dict[str, float]:
    """{part id: its share of its suite's seconds} for both test steps' ids of each suite ``splits`` names."""
    return {f"{job}:{name}:{k}/{v['m']}": x for name, v in splits.items() for k, x in enumerate(v["shares"], 1)
            for job in SHARDS.SUITE_JOBS}


def audit_provers(packages: list[str]) -> dict[str, list[str]]:
    """{`lean-audit:PKG`: its provers' audits} for each of ``packages`` whose `proved_in` names some (`lean_audit.provers`): its
    audit audits them beside it, from one build of theirs, so a plan puts them on one shard."""
    la = _lean_audit()
    return {f"lean-audit:{p}": [f"lean-audit:{q}" for q in qs] for p in packages if (qs := la.provers(p))}


def shard_needs() -> dict[str, list[str]]:
    """What a shard runs beside a part (`shards.builds`): each job's `needs` and `after`, and each audited package's provers
    (`audit_provers`), which `shards.units` holds on its shard."""
    return {**{j.name: [*j.needs, *([j.after] if j.after else [])] for j in JOBS}, **audit_provers(_parts_of()["lean-audit"])}


def given_suites(suites: dict[str, float], given: dict[str, float] | None) -> dict[str, float]:
    """``suites`` (each suite's seconds) with a suite's whole part's in ``given`` ({part: seconds}) over them."""
    out = dict(suites)
    for p, s in (given or {}).items():
        job, name, spec = SHARDS.suite_part(p)
        if job in SHARDS.SUITE_JOBS and name and spec is None:
            out[name] = float(s)
    return out


def auto_plan(costs: tuple, max_shards: int = SHARDS.MAX_SHARDS, deferred: str | None = None,
              n: int | None = None, given: dict[str, float] | None = None, split_s: float = SHARDS.SPLIT_S) -> dict:
    """The auto plan (`shards.plan`) of this tree's parts (`auto_parts`: of the jobs on the gate, or with ``deferred``, of those
    steps and what they need) at ``costs`` (`measure`'s suites, audits, steps and test files' seconds, `shards.priced`),
    ``given``'s {part: seconds} over them, read as steps' seconds too (a job's whole, spread over its parts); each suite past
    ``split_s`` split by test file (`suite_splits`)."""
    jobs = DECLARATION.needed(JOBS, [s for s in deferred.split(",") if s]) if deferred else None
    given = {p: float(s) for p, s in (given or {}).items()}
    suites = given_suites(costs[0], given)
    splits = suite_splits(suites, costs[3] if len(costs) > 3 else {}, split_s)
    names = auto_parts(deferred is None, jobs, splits={k: v["m"] for k, v in splits.items()})
    measured = {**SHARDS.priced(names, suites, costs[1], {**costs[2], **given}, split_shares(splits)),
                **{p: s for p, s in given.items() if p in names}}
    kept = {k: {f: v[f] for f in ("m", "seconds", "digest", "timings")} for k, v in splits.items()
            if any(SHARDS.suite_part(p)[1] == k for p in names)}
    return SHARDS.plan(SHARDS.tasks(names, measured), shard_needs(), max_shards, n, kept)


def _cost_json(f: Path, skipped: list[str]):
    """``f``'s JSON, or None when it can't be read, said in ``skipped``."""
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError) as e:
        skipped.append(f"{f}: {type(e).__name__}")
        return None


def measured_costs(check_roots: list[Path], test_roots: list[Path],
                   skipped: list[str] | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """(each suite's seconds, each Lake package's audit seconds) in these verdict caches: a suite's newest kept pass (its
    `seconds`), else a `durations.json` of one of ``test_roots`` (`suites.py`'s last run there); a package's newest passing audit
    with the kernel replay (its report's `seconds.total`).  A file that can't be read prices nothing and is named in
    ``skipped``."""
    skipped = [] if skipped is None else skipped
    suites: dict[str, float] = {}
    when: dict[str, str] = {}
    for root in test_roots:
        for f in [*root.glob("*.json"), *root.glob(f"{SUITES.PER_TEST}/*/{SUITES.WHOLE}--*.json")]:
            e = _cost_json(f, skipped)
            if (isinstance(e, dict) and isinstance(e.get("suite"), str) and isinstance(e.get("seconds"), (int, float))
                    and e["seconds"] > 0 and str(e.get("when") or "") >= when.get(e["suite"], "")):
                # a part's pass (`suites.py --files`) prices its suite at its seconds over its share of the suite's
                share = (e.get("part") or {}).get("share", 1.0) if isinstance(e.get("part"), dict) else 1.0
                if isinstance(share, (int, float)) and share > 0:
                    suites[e["suite"]], when[e["suite"]] = float(e["seconds"]) / share, str(e.get("when") or "")
    for root in test_roots:
        durations = _cost_json(root / "durations.json", skipped) if (root / "durations.json").is_file() else None
        for s, v in durations.items() if isinstance(durations, dict) else ():
            if s not in suites and isinstance(v, (int, float)) and v > 0:
                suites[s] = float(v)
    audits: dict[str, float] = {}
    at: dict[str, float] = {}
    for root in check_roots:
        for f in (root / "lean-audit").glob("*.json"):
            doc = _cost_json(f, skipped)
            rep = doc.get("report") if isinstance(doc, dict) else None
            pkg, sec = (rep.get("package"), rep.get("seconds")) if isinstance(rep, dict) else (None, None)
            if not (isinstance(pkg, str) and isinstance(sec, dict) and "replay" in sec and not rep.get("failures")
                    and isinstance(sec.get("total"), (int, float))):
                continue
            if (mtime := f.stat().st_mtime) >= at.get(pkg, -1.0):
                audits[pkg], at[pkg] = float(sec["total"]), mtime
    return suites, audits


def recorded_steps(store, rids: list[str], skipped: list[str] | None = None) -> dict[str, float]:
    """The seconds each step took in the newest of the recorded runs ``rids`` (newest first) where it passed, from their results'
    steps: under the step's name, or under its part when it ran one part (a shard's), circuit-check's spread over its parts.  A
    run whose result can't be read prices nothing and is named in ``skipped``."""
    skipped = [] if skipped is None else skipped
    out: dict[str, float] = {}
    for rid in rids:
        try:
            art = ((store.get_attempt(rid) or {}).get("outputs") or {}).get("result")
            meta = store.get_manifest(art).meta if art else None
        except Exception as e:  # noqa: BLE001  (a run whose result can't be read prices nothing)
            skipped.append(f"{rid}: {type(e).__name__}: {e}")
            continue
        for s in (meta or {}).get("steps") or [] if isinstance(meta, dict) else []:
            sec, ps = s.get("seconds"), s.get("parts") or []
            if s.get("status") != "passed" or not isinstance(sec, (int, float)) or sec <= 0:
                continue
            if len(ps) == 1:
                out.setdefault(ps[0], float(sec))
            elif not ps:
                out.setdefault(s["step"], float(sec))
            elif s["step"] == "circuit-check":
                for p in ps:
                    out.setdefault(p, float(sec) / len(ps))
    return out


def measure(d: Path, gather: bool = True, store=None, split_s: float = SHARDS.SPLIT_S,
            given: dict[str, float] | None = None) -> tuple[tuple[dict, dict, dict, dict], list[str]]:
    """((suites, audits, steps, times), notes): the costs the newest recorded checks measured, their verdict packs
    (`gathered_packs`, unpacked under ``d``) and their results' step seconds (`recorded_steps`), unless not ``gather``, and this
    machine's caches (`measured_costs`); and for each suite past ``split_s`` (``given``'s seconds over the measured), its test
    files' recorded seconds (`suites.file_times`, from every per-test record those hold)."""
    check_roots = [cache_dir()]
    test_roots = [VERDICTS.roots()["tests"], *sorted((Path.home() / ".cache" / "verity").glob("tests-slot-*"))]
    steps: dict[str, float] = {}
    notes, skipped = [], []
    if gather:
        try:
            from research.store.local import LocalStore

            store = store or LocalStore()
            packs = gathered_packs(d, store)
            for i, (_rid, pack) in enumerate(packs):
                roots = {"check": d / f"pack-{i}" / "check", "tests": d / f"pack-{i}" / "tests"}
                was = VERDICTS.roots
                VERDICTS.roots = lambda r=roots: r
                try:
                    VERDICTS.import_(pack)
                finally:
                    VERDICTS.roots = was
                check_roots.insert(0, roots["check"])
                test_roots.insert(0, roots["tests"])
            steps = recorded_steps(store, [rid for rid, _ in packs], skipped)
            notes.append(f"costs from the {len(packs)} newest recorded checks ({', '.join(rid for rid, _ in packs)}) and this "
                         "machine's caches")
        except Exception as e:  # noqa: BLE001  (no store: price from this machine's caches, and say so)
            notes.append(f"no recorded checks' costs ({type(e).__name__}: {e}): this machine's caches alone")
    suites, audits = measured_costs(check_roots, test_roots, skipped)
    long = {s.name: s for s in SUITES.discover() if (given_suites(suites, given).get(s.name) or 0) > split_s}
    times = {n: SUITES.file_times(s, [r / SUITES.PER_TEST / s.name.replace("/", "_") for r in test_roots]) for n, s in long.items()}
    costs = (suites, audits, steps, times)
    if skipped:
        notes.append(f"{len(skipped)} unreadable cost source(s) priced nothing: {', '.join(skipped[:3])}")
    return costs, notes


def shard_plan_main(argv: list[str]) -> int:
    """`--shard-plan`: print the auto plan of this tree (`auto_plan`), running nothing."""
    import argparse

    ap = argparse.ArgumentParser(prog="check.py --shard-plan", description="the auto plan of this tree, running nothing")
    ap.add_argument("--max-shards", type=int, default=SHARDS.MAX_SHARDS, help="the most shards (default %(default)s)")
    ap.add_argument("--n", type=int, help="this many shards, not the planner's N")
    ap.add_argument("--deferred", metavar="STEP,...", help="plan these steps and what they need (drive's after-landing run)")
    ap.add_argument("--costs", type=Path, help="a JSON {part: seconds} over the measured costs")
    ap.add_argument("--split-s", type=float, default=SHARDS.SPLIT_S, metavar="S",
                    help="split a suite measured past S seconds by test file (default %(default)s)")
    ap.add_argument("--no-gather", action="store_true", help="price from this machine's caches alone, not the recorded checks")
    ap.add_argument("--json", action="store_true", help=f"print the plan as {SHARDS.KIND}")
    ap.add_argument("--out", type=Path, help="also write the plan here")
    a = ap.parse_args(argv)
    try:
        given = json.loads(a.costs.read_text()) if a.costs else None
    except (OSError, ValueError) as e:
        print(f"check: --shard-plan: --costs: {e}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="check-shard-plan-") as d:
        costs, notes = measure(Path(d), not a.no_gather, split_s=a.split_s, given=given)
    try:
        doc = {**auto_plan(costs, a.max_shards, a.deferred, a.n, given, a.split_s), "notes": notes}
    except (DECLARATION.DeclarationError, ValueError, OSError) as e:
        print(f"check: --shard-plan: {e}", file=sys.stderr)
        return 2
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1) if a.json else "\n".join([*(f"shard plan: note: {x}" for x in notes), *SHARDS.lines(doc)]))
    return 0


def auto_shard(spec: str, gated: bool = True, plan_file: Path | None = None, jobs: set[str] | None = None) -> dict:
    """`--shard auto:K/N`: {"k", "n", "parts": shard K's parts of the auto plan, "all": the plan's parts, "plan": its summary}.
    The plan is ``plan_file`` (in a recorded run, its `AUTO_PLAN` input); DeclarationError when there is none, or its parts aren't exactly this tree's: every part of the jobs on the gate (not ``gated``: of every job), of
    ``jobs`` alone when given (an after-landing run's steps and what they need)."""
    m = AUTO_SHARD.fullmatch(spec)
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise DECLARATION.DeclarationError(f"--shard {spec}: an auto shard is auto:K/N, K in 1..N")
    k, n = int(m.group(1)), int(m.group(2))
    path = plan_file if plan_file is not None and plan_file.is_file() else None
    if path is None:
        raise DECLARATION.DeclarationError(f"--shard {spec}: no auto plan ({AUTO_PLAN} among the run's inputs, or "
                                           "--shard-plan-file): `--record --shard auto` plans once and launches every shard with it")
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise DECLARATION.DeclarationError(f"--shard {spec}: {path} can't be read ({type(e).__name__}: {e})") from None
    splits = doc.get("splits") if isinstance(doc, dict) and isinstance(doc.get("splits"), dict) else {}
    want = auto_parts(gated, jobs, splits={s: v["m"] for s, v in splits.items()})
    if why := SHARDS.check(doc, n, want):
        raise DECLARATION.DeclarationError(f"--shard {spec}: {path.name} is no auto plan of this tree: {why}")
    return {"k": k, "n": n, "parts": doc["shards"][k - 1]["parts"], "all": want, "plan": SHARDS.summary(doc),
            **({"splits": splits} if splits else {})}


def suite_parts() -> dict[str, int]:
    """{suite: P} for each suite of this tree that declares `[tool.verity.tests] parts = P`: its P parts by test id
    (`suites.id_part_of`) are parts of the check's plans.  DeclarationError when a suite declares another `parts`."""
    try:
        return {s.name: s.parts for s in SUITES.discover() if s.parts}
    except ValueError as e:
        raise DECLARATION.DeclarationError(str(e)) from None


def _parts_of() -> dict[str, list[str]]:
    """The split jobs' parts on this tree (`declaration.job_parts`): lean-audit's Lake packages, lean-suites' suites (`AFTER_LEAN`),
    each one in `suite_parts` as its parts `SUITE:K/P`."""
    files = tracked_files(ROOT)
    if files is None:
        raise DECLARATION.DeclarationError("this tree's files can't be listed, so its Lake packages aren't known")
    by = suite_parts()
    return {"lean-audit": _lean_audit().packages(files),
            "lean-suites": [x for s in AFTER_LEAN for x in ([f"{s}:{k}/{by[s]}" for k in range(1, by[s] + 1)] if s in by else [s])]}


def shard_parts() -> list[str]:
    """Every part an unsharded check covers on this tree (`declaration.all_parts`)."""
    return DECLARATION.all_parts(JOBS, _parts_of())


#: the jobs off the merge gate (drive's policy, policy.toml's `off_gate`), which a recorded check leaves to drive's after-landing run
OFF_GATE = tuple(j.name for j in JOBS if not j.gate)


def gated_parts() -> list[str]:
    """Every part of the jobs on the merge gate (`declaration.gated`): what a recorded check runs, unsharded or as a plan's
    shards together, and what `research merge` requires of it."""
    return DECLARATION.all_parts(DECLARATION.gated(JOBS), _parts_of())


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
    if not why:
        why = circuit_join(store, done, run_dir or Path(tempfile.mkdtemp(prefix="check-combine-")))
    if not why:
        why = suite_join(store, done, Path(tempfile.mkdtemp(prefix="check-combine-suites-")))
    if run_dir is not None:
        doc = {"schema": "research/result/v0.1", "kind": "check/v1", "run_id": SETTINGS.require("RESEARCH_RUN_ID"),
               "settings": SETTINGS.record(), "workload_fingerprint": {"steps": [s["step"] for s in done or []], "shards": ids},
               "validation": {"status": "failed" if why else "passed", "checker": "tools/verity/check/check.py --combine",
                              "detail": why or f"{len(ids)} shards of {commit[:12]}"},
               "steps": done or [], "shards": ids,
               **({"artifacts": ["circuit-check.json"]} if (run_dir / "circuit-check.json").exists() else {})}
        (run_dir / "result.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"check: --combine {' '.join(ids)}: " + (f"refused: {why}" if why else f"a passing check of {commit[:12]} "
                                                    f"({len(done)} steps)"), flush=True)
    return 1 if why else 0


def circuit_join(store, steps: list[dict], into: Path) -> str | None:
    """Why the circuit-check parts of a set of shards (``steps``, `shard_set`'s, each part with the run that ran it), joined, are
    not the passing `--all` each part's own pass makes them look, or None.  Joined only when they ran in more than one shard
    (one that holds every part runs `--all` whole): each shard's `CC_SHARDS`, from its Attempt's run files into ``into``, goes
    to `circuit-check --combine` (`circuit_combine`, its report ``into``'s circuit-check.json), which refuses shards that are not
    every one of one list exactly and otherwise fails as the unsharded run does."""
    held: dict[str, list[str]] = {}
    for s in steps:
        job, _, part = s["step"].partition(":")
        if job == "circuit-check" and part:
            held.setdefault(s["run_id"], []).append(part)
    if len(held) < 2:
        return None
    dirs = []
    for rid, parts in held.items():
        try:
            files = ((store.get_attempt(rid) or {}).get("outputs") or {}).get("run_files")
            if not files:
                return f"circuit-check: {rid} published no run files, so its shards' records can't be joined"
            (into / rid).mkdir(parents=True, exist_ok=True)
            got = Path(store.fetch(files, to=into / rid, paths=[CC_SHARDS]))
        except Exception as e:  # noqa: BLE001  (a guard never fails open)
            return f"circuit-check: {rid}'s shards' records can't be read ({type(e).__name__}: {e})"
        dirs += [got / CC_SHARDS / p.partition("/")[0] for p in parts]
    r = circuit_combine(dirs, into / "circuit-check.json")
    if r.returncode == 0:
        return None
    said = [line for line in (r.stderr + r.stdout).splitlines() if line.strip()][-3:]
    return (f"circuit-check --combine of its {len(dirs)} shards {'refused them' if r.returncode == 2 else f'exited {r.returncode}'}"
            + (f": {' / '.join(said)}" if said else ""))


def suite_join(store, steps: list[dict], into: Path) -> str | None:
    """Why the parts of the suites a set of shards split (``steps``, `shard_set`'s: `pytest:SUITE:K/M` by test file,
    `lean-suites:SUITE:K/P` by test id, each with the run that ran it) don't make each suite's passing verdict, or None: each
    run's suite reports (its `suites.json` and `suites-lean.json`), from its Attempt's run files into ``into``, joined
    (`suites.combine_parts`), must hold every part of one partition once, each passed, so a part run on another partition (a
    machine with another timing record) passes nothing; or the suite's own pass, from the shard that held every part and ran it
    whole."""
    runs = sorted({s["run_id"] for s in steps if SHARDS.suite_part(s["step"])[2] and s.get("run_id")})
    if not runs:
        return None
    split = {SHARDS.suite_part(s["step"])[1] for s in steps if SHARDS.suite_part(s["step"])[2]}
    rows = []
    for rid in runs:
        try:
            files = ((store.get_attempt(rid) or {}).get("outputs") or {}).get("run_files")
            if not files:
                return f"suites: {rid} published no run files, so its suites' parts can't be joined"
            names = [n for n in ("suites.json", "suites-lean.json") if any(f["path"] == n for f in store.listing(files))]
            got = Path(store.fetch(files, to=into / rid, paths=names))
            rows += [r for n in names for r in json.loads((got / n).read_text())
                     if isinstance(r.get("part"), dict) or r.get("name") in split]
        except Exception as e:  # noqa: BLE001  (a guard never fails open)
            return f"suites: {rid}'s suite reports can't be read ({type(e).__name__}: {e})"
    joined = {r["name"]: r for r in SUITES.combine_parts(rows)}
    bad = [f"{n}: {'; '.join(joined[n].get('why') or [str(joined[n].get('status'))])}" if n in joined
           else f"{n}: no shard reported its parts" for n in sorted(split)
           if n not in joined or joined[n]["status"] not in ("passed", "cached")]
    return f"split suites don't make their suites' verdicts: {' / '.join(bad)}" if bad else None


def circuit_combine(dirs: list[Path], out: Path) -> subprocess.CompletedProcess:
    """`circuit-check --combine` of the shard directories ``dirs``, its report to ``out``: exit 0 for the unsharded run's pass,
    1 for its failure, 2 when it refuses the shards."""
    r = subprocess.run([sys.executable, "-m", "circuit_check", "--combine", *map(str, dirs), "--out", str(out)], cwd=ROOT,
                       capture_output=True, text=True)
    print(r.stdout + r.stderr, end="", flush=True)
    return r


#: What reaches the steps from this process's environment, Turborepo's strict mode being the model: where things are (PATH, HOME,
#: the temporary and XDG directories, the Rust and Lean toolchains' homes), the locale, how this machine reaches the network (its
#: proxies and certificate bundles), uv and the environment it runs in, the GPUs the run may use (`research run` hides those
#: gpu-lease hands out), and the switches tests read (`VERITY_*`, `FLOCK_*`).
#: Nothing else: not the store's, RunPod's or GitHub's credentials, not what the recording run gives its workload (`RESEARCH_*`,
#: the machine's PYTHONPYCACHEPREFIX; the tests start runs of their own, which would inherit them).
ENV_PASS = frozenset({"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LANGUAGE", "VIRTUAL_ENV", "CARGO_HOME", "RUSTUP_HOME",
                      "ELAN_HOME", "SSL_CERT_FILE", "SSL_CERT_DIR", "http_proxy", "https_proxy", "no_proxy", "all_proxy",
                      "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "CUDA_VISIBLE_DEVICES"})
ENV_PASS_PREFIXES = ("LC_", "XDG_", "UV_", "VERITY_", "FLOCK_")
#: What check sets for its steps from its own state, never from the environment: a check run inside a step (its own tests) would
#: otherwise pass its parent's on.
ENV_OWN = frozenset({CUSTODY_ENV, STORE_REQUIRED_ENV, REGRESSION_ENV})


def env() -> dict[str, str]:
    """The steps' environment: this process's variables that `ENV_PASS` and `ENV_PASS_PREFIXES` name, less `ENV_OWN`; PYTHONPATH
    entries inside this tree only (a machine's runner puts its shipped `research` first); and a git identity.  Inside a recorded
    run, the store is required, the run's custody dir is passed on and the vLLM reference regression runs (the module docstring's
    "The evidence store")."""
    out = {k: v for k, v in SETTINGS.environ().items() if (k in ENV_PASS or k.startswith(ENV_PASS_PREFIXES)) and k not in ENV_OWN}
    if SETTINGS.RESEARCH_RUN_ID:
        out[STORE_REQUIRED_ENV] = "1"
        out[REGRESSION_ENV] = "1"
        if custody := custody_dir():
            out[CUSTODY_ENV] = str(custody)
    keep = [p for p in SETTINGS.PYTHONPATH.split(os.pathsep) if p and Path(p).resolve().is_relative_to(ROOT)]
    if keep:
        out["PYTHONPATH"] = os.pathsep.join(keep)
    else:
        out.pop("PYTHONPATH", None)
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):          # tests commit in throwaway repos; a fresh machine has no identity
        out.setdefault(k, "check")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        out.setdefault(k, "check@localhost")
    return out


#: what a step's key holds of its environment beside the switches tests read (`SUITES.keyed_variables`): the locale's, which say
#: what encoding text without a declared one is read in, and the tree's own PYTHONPATH entries (`env`), which say what is imported
KEYED_ENV = ("LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH")


def _env_key(e: dict[str, str]) -> dict[str, str]:
    """The environment as a key: what a step's verdict can turn on (`KEYED_ENV`, `SUITES.keyed_variables`), with this tree's path
    written `<tree>` (a machine keeps each commit's tree at its own path). Not where a machine keeps things (PATH, HOME, the
    temporary and XDG directories, uv's and the toolchains' homes, a check slot's test cache), how it reaches the network or what
    names it, which differ between nodes for the same inputs: what a step runs that the PATH finds is keyed by itself (the
    interpreter, `rustc -vV`, the Lean toolchain's pin in the tree)."""
    keep = {k: v for k, v in e.items() if k in KEYED_ENV} | SUITES.keyed_variables(e)
    return {k: v.replace(str(ROOT), "<tree>") for k, v in sorted(keep.items())}


def agreement_key(e: dict[str, str]) -> str | None:
    """What the agreement job reads: the tracked files `agreement_closure` names (verity/protocols/flock/agreement/: `ci.py` and the scripts
    it runs, which import only the standard library and each other, `upstream.json` pinning the upstream build,
    the inputs and the job's settings; and the Lean packages `flock-verify` is built from, verity/protocols/flock/verifier/ and
    the packages of the tree it requires by path, their sources and toolchain pins), the interpreter and the platform.  None
    when the tree's files cannot be listed."""
    files = tracked_files()
    if files is None:
        return None
    h = hashlib.sha256(json.dumps([sys.version, SUITES.platform_id(), _env_key(e)]).encode())
    for f in agreement_closure(files):
        p = ROOT / f
        h.update(f.encode() + b"\0" + (_sha256(p).encode() if p.is_file() else b"-") + b"\0")
    return h.hexdigest()


def rust_tests_inputs(files: list[str]) -> list[str]:
    """What `check_build.sh test` reads of the tree: the script (which pins the Flock commit and the order the patches go on in),
    the patches, and the live crate it links in."""
    pre = "verity/protocols/flock/prover/"
    return [f for f in files if f == pre + "check_build.sh" or f.startswith(pre + "live/")
            or (f.startswith(pre + "flock-") and f.endswith("-b684b12.patch"))]


def rust_toolchain() -> str | None:
    """The Rust toolchain `check_build.sh` pins: its `RUST_TOOLCHAIN` default (check passes no `RUST_TOOLCHAIN` of its own)."""
    p = ROOT / "verity" / "protocols" / "flock" / "prover" / "check_build.sh"
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
    h = hashlib.sha256(json.dumps([r.stdout, SUITES.platform_id(), _env_key(e)]).encode())
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
    return hashlib.sha256(json.dumps([tree.stdout.strip(), list(LINTS), sys.version, SUITES.platform_id(), _env_key(e)]).encode()).hexdigest()


#: the steps whose pass is kept under a key of everything they read, and reused while it holds
STEP_KEYS = {"lean-agreement": agreement_key, "flock-rust-tests": rust_tests_key, "preflight-lints": lints_key}


def agreement_closure(files: list[str]) -> list[str]:
    """The tracked files the agreement job can read: verity/protocols/flock/agreement/, the verifier's Lean package and the
    packages of the tree it requires by path (`required_in_tree`), which its executable is built from; less any Lake package
    nested in the verifier's (Flock's proofs are under `security/Proofs/`), which it is not built from as long as the Lean
    package requires nothing (a `require` in its lakefile keeps them all in), and less each of those packages'
    `lean-audit.json`, which only `tools/verity/lean/check.py` reads (a test fails if the agreement's scripts or the
    verifier's lakefile mention it)."""
    pre, lean = "verity/protocols/flock/agreement/", "verity/protocols/flock/verifier/"
    lakefiles = [f for f in files if f.startswith(lean) and f.rsplit("/", 1)[-1] in ("lakefile.toml", "lakefile.lean")]
    root = [f for f in lakefiles if f.count("/") == lean.count("/")]
    nested = tuple(f.rsplit("/", 1)[0] + "/" for f in lakefiles if f not in root)
    if any("require" in (ROOT / f).read_text() for f in root if (ROOT / f).is_file()):
        nested = ()
    packages = (lean, *required_in_tree(lean))
    locks = {p + "lean-audit.json" for p in packages}
    return [f for f in files if f.startswith((pre, *packages)) and f not in locks and not f.startswith(nested or ("\0",))]


#: the repository-wide lints the preflight runs: each fails a whole check, and together they take seconds
LINTS = ("tests/test_no_wall_clock.py", "tests/test_repository.py")
#: how long a stopped step's processes have after SIGINT before SIGKILL
STOP_GRACE_S = 60
#: each step's record, one JSON line appended as the step ends
STEPS_LOG = "steps.jsonl"


def preflight() -> list[tuple[str, list[str] | str, Path]]:
    """The steps before the groups: `uv lock --check` (a current lock needs no index), then `LINTS` (without bytecode or
    pytest's cache, so they leave nothing in the tree), then `fetchable_fixtures`. Under `uv run`, `$UV` names uv."""
    uv = SETTINGS.UV or shutil.which("uv")
    lock = [uv, "lock", "--check"] if uv else "failed: no `uv` here to check `uv.lock` (run check through `uv run`)"
    return [("preflight-lock", lock, ROOT),
            ("preflight-lints", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *LINTS], ROOT),
            ("preflight-fixtures", [sys.executable, "-B", str(Path(__file__).resolve()), "--fixtures"], ROOT)]


#: preflight steps that judge the machine the check runs on, so `--preflight` elsewhere (a train's prep) only reports them
ON_NODE = ("preflight-fixtures",)


def fetchable_fixtures(root: Path = ROOT, url: str | None = None, head=None, store=None) -> tuple[dict[str, str], list[str]]:
    """({id: where}, [id]): where each fetched fixture (a `fixtures/artifacts.json` entry whose path git doesn't hold) can come
    from on this machine, and those that can come from nowhere: the public export, asked for its manifest only (one HEAD, no
    blob), else the tree already holding its bytes, else this machine's store holding its manifest and every blob.  A fixture
    preserved privately and never exported is in neither the public export nor a node's own store, so a check there fails
    after its launch; a copy put in the node's store keeps a train moving before the export.  Raises when the public export
    is needed and there is none."""
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
#: lean-suites failed, holding its train slot, and circuit-check beside it its share of memory; a passing step's longest
#: quiet stretch is about an hour (circuit-check's plan before its first target, one 35-minute target, a cold Lean package's
#: build)
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


#: the steps whose failure stops the run even under --keep-going when their log has this text (`halts_on` in ci.toml); without
#: it (a failed clone or toolchain download) the other groups keep their verdicts
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

    def __init__(self, out: Path, before: dict[str, str], records=None):
        self.out, self.before, self.records = out, before, records
        self.lock = threading.Lock()
        self.want = threading.Event()
        self.closed = False
        self.written: tuple | None = None
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
        """Write the pack unless it holds what is new already; the number of entries it holds.  With ``records`` (a
        `records.Publisher`), the circuit records it names are put on the store first, except once a SIGTERM came (its SIGKILL
        follows in seconds), and those the store holds are named in the pack in place of their bytes."""
        with self.lock:
            try:
                new = VERDICTS.changed(self.before)
                drop, manifest = set(), None
                if self.records is not None:
                    try:
                        drop, manifest = self.records.pack(publish=not TERMINATED)
                    except Exception as e:  # noqa: BLE001  (the records then go in the pack as they are)
                        print(f"check: circuit records: naming them failed ({type(e).__name__}: {e}); the pack holds them",
                              flush=True)
                state = ({k: d for k, (d, _) in new.items()}, manifest)
                if state != self.written:
                    self.entries = VERDICTS.export(self.out, self.before, new, drop, (manifest,) if manifest else ())
                    self.written = state
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
    They are what its plan of its own tree (`plan.py --own`: this run's caches, the packs imported) finds passed, of the
    entries here before its steps ran (``before``).  The plan runs once the twins are built (`run`'s `prebuild`, which no other
    process may import while it compiles them), beside the steps; it runs again once circuit-check ends when it found no task
    list to read circuit-check's targets from.  xz with a dictionary spanning many entries (`verdicts.XZ_FILTERS`): one tree's
    circuit-check entries run past a GB, and pack to a few MB.  `verdict_packs` never takes it; `plan.py` does, for a check
    of the commit it plans and each `--from`, and so does an uncached check of that commit (`commit_packs`).  With ``only``
    (`--no-cache --circuit-cache`), just its entries under that pack path: the plan also finds suites' passes here, which such
    a run runs again."""

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
    and a step `HALTS_ON` marks `halts` stops them even then), and each step's record appended to `STEPS_LOG`
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
            self.spans.open(name)
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
            if rec["status"] == "failed" and not self.stopping and (not self.keep_going or rec.get("halts")):
                self.halted = rec["step"]
                self.stop(f"{rec['step']} failed")
        return rec


def _step(name: str, argv: list[str] | str, cwd: Path, out: Path, cache: bool, steps: Steps) -> dict:
    if isinstance(argv, str):
        skipped = argv.startswith("skipped:")
        print(f"check: {name}: {argv}", flush=True)
        return steps.end({"step": name, "status": "skipped" if skipped else "failed", "detail": argv, "seconds": 0.0})
    e = env() if cache else env() | SUITES.UNCACHED_ENV
    t0 = time.monotonic()
    hit = None
    key = STEP_KEYS[name](e) if cache and name in STEP_KEYS else None
    if name == "pytest" and cc_manifests():
        e = e | {RECORDS.SESSION_ENV: str(out / "circuit-records.session.json")}
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
            # handler marks it stopped: it found nothing (it would otherwise read as "failed" with an empty log)
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
PREBUILD = "import json\nfrom verity.definitions.numerics import twins\nprint(json.dumps(twins.preload()))\n"
#: the prebuild's output, in the output directory beside the steps' logs
PREBUILD_LOG = "prebuild.log"


def prebuild(out: Path, steps: Steps) -> dict | None:
    """`PREBUILD` in a process of its own, its output in `PREBUILD_LOG`: None once it built every model, else its record, a
    `prebuild` step that failed with its exit status (or was stopped by the run's stop), which refuses the run: no group
    starts, since each would compile the models the first time it ran one."""
    t0 = time.monotonic()
    log = out / PREBUILD_LOG
    with open(log, "wb") as f:
        rc = subprocess.run([sys.executable, "-c", PREBUILD], cwd=ROOT, env=env(), stdout=f, stderr=subprocess.STDOUT).returncode
    seconds = round(time.monotonic() - t0, 1)
    if rc == 0:
        print(f"check: prebuild: the C++ models are built in {seconds} s ({PREBUILD_LOG})", flush=True)
        return None
    if steps.stopping or rc == -signal.SIGTERM:
        return steps.end({"step": "prebuild", "status": "stopped", "rc": rc, "log": PREBUILD_LOG, "seconds": seconds,
                          "detail": f"stopped: {steps.why or 'killed by a SIGTERM this check did not send'}"})
    tail = log.read_bytes().decode(errors="replace").splitlines()
    print("\n".join(f"  prebuild| {line}" for line in tail[-40:]), flush=True)
    why = (f"refused: the C++ models' prebuild exited {rc} ({(tail[-1:] or ['it wrote nothing'])[0].strip()}; {PREBUILD_LOG}), "
           "so a step would compile them untraced")
    print(f"check: prebuild: {why}", flush=True)
    return steps.end({"step": "prebuild", "status": "failed", "rc": rc, "log": PREBUILD_LOG, "seconds": seconds, "detail": why})


def run(out: Path, n: int | None = None, cache: bool = True, keep_going: bool = False,
        only: list[str] | None = None, packs: Packs | None = None, reused: Reused | None = None,
        regenerate: bool = True, circuit_cache: bool | None = None, live=None,
        splits: dict[str, dict] | None = None, bookkeeping: bool = False) -> tuple[bool, list[dict]]:
    n = n or cores()
    urls, agreement_urls = take_lean_deps_urls(out), take_agreement_urls(out)
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
        if not steps.stopping and all(r["status"] == "passed" for r in results) and (refused := prebuild(out, steps)):
            results.append(refused)
        if not steps.stopping and all(r["status"] == "passed" for r in results):
            if reused is not None:
                reused.start()
            gs = groups(out, n, cache, urls, only=only, regenerate=regenerate, circuit_cache=circuit_cache, splits=splits,
                        bookkeeping=bookkeeping, agreement_urls=agreement_urls)
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
        for taken in (urls, agreement_urls):
            if taken and taken.name.startswith(TAKEN):
                taken.unlink(missing_ok=True)
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
    or a test whose outcome isn't a function of its inputs), among the slot's own and, in a recorded run, every pass the commit's
    recorded checks hold (`commit_packs`, imported before the steps); ``nondeterministic``, the tests that both passed and failed
    under one key (their inputs were identical)."""
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
    config (the tree of `tools/verity/check/` at HEAD) and each of its steps' parts (the plan's names that step ran); for an
    after-landing run (`--deferred`), `DEFERRED_KIND`, the same under `deferred`.  Its `passes` are the lean-audit step's
    (`lean_passes`), which the store indexes as `passes.<key>`.  A run ``stopped_by`` a signal is `not_run`: no verdict and
    no passes, so a job queue runs it again."""
    if shard is not None:
        done = [{**s, "parts": got} if (got := [p for p in shard["parts"] if p.partition(":")[0] == s["step"]]) else s
                for s in done]
        shard = {**shard, "commit": _rev("HEAD"), "config": _rev("HEAD:tools/verity/check")}
    deferred = bool(shard and shard.get("deferred"))
    kind = DEFERRED_KIND if deferred else SHARD_KIND if shard else "check/v1"
    doc = {"schema": "research/result/v0.1", "kind": kind, "run_id": SETTINGS.require("RESEARCH_RUN_ID"),
           "reuse": reuse(run_dir, done), "settings": SETTINGS.record(), "workload_fingerprint": {"steps": [s["step"] for s in done]},
           "validation": {"status": "not_run" if stopped_by else "passed" if ok else "failed", "checker": "tools/verity/check/check.py",
                          "detail": (f"stopped by {stopped_by}: " if stopped_by else "")
                                    + "; ".join(f"{s['step']} {s['status']}" for s in done)},
           "measurements": [{"name": f"{s['step']}_wall", "value": s["seconds"], "unit": "s"} for s in done],
           "passes": {} if stopped_by else lean_passes(run_dir, done),
           "steps": done, "artifacts": [p for p in ("suites.json", "suites-lean.json", "suites/", "circuit-check.json", f"{CC_SHARDS}/", "lean-audit/",
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
    """The run's `LIVE.Live`: what it checks (a shard, an after-landing run's steps, parts, or the whole check) of which commit, published
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


def resync(argv: list[str]) -> None:
    """Install the workspace into this fresh environment (`SUITES.SYNC`) and run check again with ``argv``, unless the sync left it
    fresh (`SUITES.fresh`), which would sync again."""
    print(f"check: {sys.prefix} holds none of the workspace yet: `{' '.join(SUITES.SYNC)}`", file=sys.stderr)
    if subprocess.run(SUITES.SYNC, cwd=ROOT).returncode == 0 and not SUITES.fresh(env()):
        os.execv(sys.executable, [sys.executable, sys.argv[0], *argv])


def main(argv: list[str], script: bool = False) -> int:
    if argv == ["--agreement-files"]:
        try:
            signed = agreement_url_file(d := Path(tempfile.mkdtemp(prefix="check-agreement-")))
        except NotSigned as e:
            shutil.rmtree(d, ignore_errors=True)
            print(f"check: --agreement-files: {e}", file=sys.stderr)
            return 1
        print(f"--send {signed}")
        return 0
    if argv == ["--lean-deps-files"]:
        urls = lean_deps_url_file(Path(tempfile.mkdtemp(prefix="check-lean-deps-")))
        print(f"--send {urls}" if urls else "")
        return 0
    if argv[:1] == ["--lean-agreement"] and len(argv) in (4, 5):
        return lean_agreement(*map(Path, argv[1:]))
    if argv[:1] == ["--lean-agreement-urls"] and len(argv) == 3:
        return lean_agreement_urls(Path(argv[1]), Path(argv[2]))
    if argv[:1] == ["--opt-in"] and len(argv) > 2:
        return opt_in(Path(argv[1]), argv[2], argv[3:])
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
    if argv[:1] == ["--shard-plan"]:
        return shard_plan_main(argv[1:])
    if "--record" in argv:
        if rid := SETTINGS.RESEARCH_RUN_ID:
            print(f"check: --record inside a recorded run ({rid}): it launches its own `research run`, which places the check "
                  f"and records it as an Attempt of the clean commit, so launch it from your checkout instead, outside `research "
                  f"run` (the launch is light; an agent VM runs it): `{nested_record(argv)}`", file=sys.stderr)
            return 2
        return record([a for a in argv if a != "--record"])
    if why := SETTINGS.refusals():
        print("\n".join(f"check: {w}" for w in why), file=sys.stderr)
        return 2
    run_dir = Path(d) if (d := SETTINGS.RESEARCH_RUN_DIR) else None
    if argv[:1] == ["--combine"] and len(argv) > 1:
        return combine(argv[1:], run_dir)
    if "--cc-records" in argv:
        if [a for a in argv if a not in ("--cc-records", "--keep-going")] or run_dir is None:
            print("check: --cc-records takes no other flag and runs inside a recorded run: `check.py --record --cc-records --on M`",
                  file=sys.stderr)
            return 2
        return cc_records(run_dir)
    cache = "--no-cache" not in argv
    circuit_cache = cache or "--circuit-cache" in argv
    keep_going = "--keep-going" in argv
    regenerate = "--no-regenerate" not in argv
    bookkeeping = "--bookkeeping-warn" in argv
    rest = [a for a in argv if a not in ("--no-cache", "--circuit-cache", "--keep-going", "--no-regenerate", "--bookkeeping-warn")]
    pack = None
    if "--verdicts-in" in rest and rest.index("--verdicts-in") + 1 < len(rest):
        i = rest.index("--verdicts-in")
        pack, rest = Path(rest[i + 1]), rest[:i] + rest[i + 2:]
    records = None
    if "--cc-records-from" in rest and rest.index("--cc-records-from") + 1 < len(rest):
        i = rest.index("--cc-records-from")
        records, rest = rest[i + 1], rest[:i] + rest[i + 2:]
    sharded = None
    plan_file = run_dir / "inputs" / AUTO_PLAN if run_dir is not None else None
    if "--shard-plan-file" in rest[:-1] and run_dir is None:
        i = rest.index("--shard-plan-file")
        plan_file, rest = Path(rest[i + 1]), rest[:i] + rest[i + 2:]
    if "--shard" in rest and rest.index("--shard") + 1 < len(rest):
        i = rest.index("--shard")
        later = rest[rest.index("--deferred") + 1] if "--deferred" in rest[:-1] else None
        try:
            sharded = shard(rest[i + 1], gated=later is None, plan_file=plan_file,
                            jobs=DECLARATION.needed(JOBS, [s for s in later.split(",") if s]) if later else None)
        except DECLARATION.DeclarationError as e:
            print(f"check: {e}", file=sys.stderr)
            return 2
        rest = rest[:i] + rest[i + 2:]
        print(f"check: shard {sharded['k']}/{sharded['n']}: {', '.join(sharded['parts'])}", flush=True)
        if p := sharded.get("plan"):
            print(f"check: shard plan {p['digest'][:12]}: N = {p['n']}, floor {p['floor_s'] / 60:.1f} min, predicted wall "
                  f"{p['predicted_s'] / 60:.1f} min, this shard {p['seconds'][sharded['k'] - 1] / 60:.1f} min", flush=True)
    splits = sharded.pop("splits", None) if sharded else None
    deferred = None
    if "--deferred" in rest and rest.index("--deferred") + 1 < len(rest):
        i = rest.index("--deferred")
        deferred, rest = rest[i + 1], rest[:i] + rest[i + 2:]
    picked = [a for a in rest if a in DECLARATION.PARTS]
    rest = [a for a in rest if a not in DECLARATION.PARTS]
    if deferred is not None:
        if picked:
            print("check: --deferred runs the post-landing steps alone, without a part", file=sys.stderr)
            return 2
        try:
            mine = deferred_parts(deferred)
        except DECLARATION.DeclarationError as e:
            print(f"check: {e}", file=sys.stderr)
            return 2
        if sharded:
            chosen = {p.partition(":")[0] for p in mine}
            if not (mine := [p for p in sharded["parts"] if p.partition(":")[0] in chosen]):
                print(f"check: --deferred {deferred} --shard {sharded['k']}/{sharded['n']}: that shard holds none of those steps",
                      file=sys.stderr)
                return 2
            sharded = {**sharded, "parts": mine}
        sharded = {**(sharded or {}), "parts": mine, "deferred": deferred.split(",")}
        print(f"check: deferred {deferred}: {', '.join(sorted({p.partition(':')[0] for p in sharded['parts']}))}", flush=True)
    if bookkeeping and (deferred is None or not regenerate or "--shard" in argv):
        print("check: --bookkeeping-warn is drive's lock run: with --deferred, unsharded and without --no-regenerate",
              file=sys.stderr)
        return 2
    if picked:
        if sharded or SETTINGS.RESEARCH_RUN_ID:
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
        if script and SUITES.fresh(env()):
            resync(argv)
        print(f"check: {sys.executable} would run this tree's steps against another checkout's code ({away}); run check with "
              f"{ROOT}'s own environment: `{' '.join(SUITES.SYNC)} && uv run --extra torch-cpu python tools/verity/check/check.py` "
              f"there", file=sys.stderr)
        return 2
    only = sharded["parts"] if sharded else None
    if only is None and OFF_GATE and cache and run_dir is not None:
        try:
            only = gated_parts()
        except DECLARATION.DeclarationError as e:
            print(f"check: {e}", file=sys.stderr)
            return 2
        print(f"check: off the merge gate, left to the after-landing run (`--deferred`): {', '.join(OFF_GATE)}", flush=True)
    out = run_dir or Path(tempfile.mkdtemp(prefix="check-"))
    listed, verdicts, named = take_verdict_urls(out), {}, {}
    if pack or listed:
        part = None if cache else ((CIRCUIT_ENTRIES, COMPARED) if circuit_cache else COMPARED)
        verdicts["in"] = [*([verdicts_in(pack, run_dir, part, named)] if pack else []),
                          *(verdicts_fetched(listed, part, named) if listed else [])]
    cc_remote = store_remote() if run_dir is not None and circuit_cache and builds_records(only) else None
    if cc_remote is not None and named:
        verdicts["records"] = records_fetched(named, cc_remote)
    if records is not None and (only is None or any(p.partition(":")[0] == "circuit-check" for p in only)):
        why = (cc_records_in(records, cache_dir() / "circuit-check") if circuit_cache
               else "circuit-check reuses no cache here (--no-cache without --circuit-cache)")
        if why:
            print(f"check: --cc-records-from {records} refused: {why}", file=sys.stderr, flush=True)
            return 2
    before = VERDICTS.snapshot() if run_dir is not None and circuit_cache else None
    t0 = time.monotonic()
    TERMINATED.clear()
    main_thread = threading.current_thread() is threading.main_thread()
    prior = signal.signal(signal.SIGTERM, _terminated) if main_thread else None
    publisher = (RECORDS.Publisher(cache_dir() / "circuit-check", CIRCUIT_ENTRIES, time.time_ns(), RECORDS.putter(cc_remote),
                                   cache_dir(), out=out) if cc_remote is not None and before is not None else None)
    packs = Packs(run_dir / "verdicts.tar.gz", before, publisher) if before is not None else None
    reused = Reused(run_dir, before, None if cache else CIRCUIT_ENTRIES) if before is not None else None
    live = live_record(out, run_dir, sharded, picked)
    live.start()
    try:
        ok, done = run(out, n, cache, keep_going, only, packs, reused, regenerate, circuit_cache, live, splits,
                       **({"bookkeeping": True} if bookkeeping else {}))
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
            if publisher is not None:
                verdicts["out"]["records"] = publisher.stats
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
    sys.exit(main(sys.argv[1:], script=True))
