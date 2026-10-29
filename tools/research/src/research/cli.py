"""`research` command line.

  research run [--project P] [--stage S] [--timeout T] [--expect K=V ...] [--require-result] [--run-record] [--runs-dir D] [harness flags] -- CMD...
  research run [--tool NAME|module:NAME] [--scratch NAME ...] [--campaign C] [...] -- CMD...   # evidence store: derivation before, attempt after
  research run --local-only REASON [...] -- CMD...   # where the store has no usable remote a local run refuses (exit 3): REASON says
                                                 # what preserves its records instead (default $RESEARCH_LOCAL_ONLY; job.json records it)
  research run --queue --source GITDIR --project P [--kind K] [--question Q] [--mem-gb G] [--disk-gb N] [--max-min M | --timeout T] [--gpus N] [--cpus N] [--class P] [--phase cpu|gpu] [--gpu-host] [--preemptible] [--quiet] [--on NODE] -- CMD...
                                                 # the one queue (infra/cluster) places the job and runs it under its node's executor
  research run --on auto [--cpus N] [--mem-gb G] [--max-min M | --timeout T] [--wait] -- CMD...   # an agent's suite or build: queued
                                                 # onto vy-nebius-2 from this checkout (kind agent-test: 4 CPUs, 16 GB, 60 min unless given;
                                                 # a suites.py run of named suites: their [tool.verity.tests] worker_gb and timeout_s),
                                                 # on whole cores of its own, waiting on the node while none are free; or, when it starts
                                                 # sooner there, a slot of a CPU node's research pool (research/auto.py's SPILL)
  research run --on MACHINE --project P [--source GITDIR] [--send FILE ...] [--env K=V ...] [--exclusive] [--mem-gb N [--mem-wait S]]
               [--gpus N] [--custody-ttl 6h | --no-custody-r2] [--preflight TOOL] [...] -- CMD...   # run custody on the store's remote is the default
                                                 # where gpu-lease hands out the GPUs, only --gpus N (a FCFS wait for N) shows the run any
                                                 # with --on; a --tool's declared preflight (or --preflight TOOL's) runs on the machine first
                                                 # it returns once the machine starts the run; --wait blocks until it ends and exits with its rc,
                                                 # for at most 20 min (--wait-max S lowers it): then exit 79, the run going on (research/waitcap.py)
  research run --job-user USER [...] -- CMD...   # the workload as USER in a sandboxed system unit; records stay the runner's (research/jobuser.py)
  research fetch <run-id> [--all] [--on NAME]    # copy the machine's records (or every file, verified) into the local run dir
  research cancel <run-id> --reason TEXT [--by WHO] [--on NAME]   # SIGTERM to the run's workload process group and its whole cgroup
                                                 # scope, recorded as a manual cancel; 20 s later the runner SIGKILLs what is left (never
                                                 # itself), records the exit and publishes custody.  From any VM: as `inspect` finds runs
  research status [RUN] [--on NAME] [--runs-dir D] [--refresh] [--balance] [--all] [--json]
  research inspect <run-id|prefix|path> [--on NAME] [--runs-dir D] [--json | --py-spy]
                                                 # a full run id another VM launched (no local dir): found on --on NAME, else on
                                                 # each ssh machine, and its records fetched here (remote.adopt)
  research keeper set NAME --on NODE --project P --n N --until UTC [--gpus G] [--max-min M] [--source GITDIR] [...] -- CMD... {seed}
  research keeper stop NAME --on NODE [--cancel] | status [NAME] --on NODE [--json]
                                                 # a standing campaign target: the node keeps N chunks leased, a fresh seed each,
                                                 # until UTC (<= 48 h), preemptible; no VM stays awake (`ops keeper`, verity_ops/keeper.py)
  research compare <original>... <replay> [--expect E] [--json]
                                                 # the replay's verdict from the store (same | within-band | differs | expected-change);
                                                 # --dirs: two run dirs' differences and ratios, which the workload contract judges
  research tele <telemetry subcommand...>        # the full harness surface (init, event, sample, cancel-intent, classify, ...)
  research data <store subcommand...>            # the evidence store: put show where fetch verify select snapshot labels label reindex sql
  research keep PATH --owner @LANE --expires DATE|never --priority low|normal|high|keep [--regenerate CMD] [--art ART | --preserve] [--why T]
  research keep show PATH                        # a node path's retention record (research/retention/v0, a sidecar), explicit or implicit
  research retention report [--root R] [--depth N] [--json]   # every recorded path and every unowned area, unowned and expired first
  research retention gc [--root R] [--force] [--high-pct 80] [--low-pct 75] [--jobs-dir D]... [--json] [--apply]
                                                 # policy default: delete only what a record allows; a dry run unless --apply (research/retention.py)
  research retention rm PATH... --approved-by @OWNER --ref REF [--apply]   # delete chosen paths with the owner's yes, which every record
                                                 # covering them must name; never a `keep`; each --apply decision goes to the deletion log
  research merge REF [--into main] [--tool check] [-m MSG] [--dry-run]   # the merge gate: `git merge --no-ff REF` only with a passing
                                                 # attempt of the tool for that exact commit in the store (research/merge.py)
  research merge --train A B C [--on MACHINE --project P]                # several refs as one candidate, one check, a fast-forward
  research queue status|sync|land|eject|ready|judge|review [--shadow] [--dry-run]  # the merge queue: ready PRs on `next`, `main`
                                                 # fast-forwarded to its reviewed prefix, checked after landing (research/queue.py)
  research jobs add|list|show|cancel|events ...  # the Job queue: batch work on pods, on the control app (research/jobs/cli.py)
  research worker --dispatch --pods A,B [--name rc]   # stage 1's dispatcher: claims jobs, runs each with `run --on` on a free pod
  research git old2new|remap|check|build ...     # after a history rewrite: an old commit's image, a clone's local work moved onto the
                                                 # rewritten history, the old-history guard, the commit-map/v1 payload (research/commitmap.py)
  research pods check-part <machine|pod-id>      # reference-part gate: name, VRAM vs [families.<fam>] vram_mb, host CPU, load, verdict (exit 3 = variant)
  research pods create <runpod flags> --max-hours H [--first-idle-min 20] [--require-reference-part] [--register --project P] [--allow-duplicate]
                                                 # only under a budget line the guard has read (exit 6); at most one live pod per --name (exit 4
                                                 # exists, 5 no stock, 2 failed); the lease armed at boot and over ssh (pods/lease.py)
  research pods extend TARGET --hours H | lease TARGET   # push the pod's lease to at least now + H / show it
  research pods register NAME --pod-id ID --project P | unregister NAME | registry [import]   # the pod registry: one file per machine (pods/registry.py)
  research pods unpreserved <machine|pod-id> [--older-min N]   # Attempts recorded on that machine that are NOT `preserved` (same predicate as `data preserved`); exit 1 if any
  research pods drain <machine|pod-id> [--force --reason TEXT]      # the report, then terminate ONLY when nothing is unpreserved (override logged)
  research pods guard [--budgets NOTES|FILE] [--dry-run] [--detach]   # the one budgets guard: every line of budgets.toml, re-read each poll
  research pods guard status|stop                # its state file / SIGTERM to exactly its recorded pid
  research env --source TREE [--rev REV] [--json]  # `export PYTHONPATH=...` from TREE's [tool.uv.workspace] members (research.pythonpath)
  research doctor [--role lane|coordinator|infra] [--json]   # can this VM do its job: secrets by name, tokens by status-only probes,
                                                 # ssh, the venv against uv.lock, notes pushes, disk, clock; exit 1 on a finding (research/doctor.py)
  research deploy drift [--on N ...] [--json] | install NODE --by WHO [--ref REF] [PATH ...] [--apply]   # what runs on the Nebius
                                                 # nodes (pods/nebius/deploy.toml) against main; installs from a pushed commit, restarting its units (`ops deploy`, verity_ops/deploy.py)
  research monitors list|check|run|relay ...     # invariants with an owner, a threshold and an action: node 1's service writes each
                                                 # incident once and its clear once; the relay posts them (research/monitors.py)
  research usage hourly [--node nebius-1|nebius-2|all] [--hours H] [--json] | wasters --day YYYY-MM-DD [--top 3] [--json]
                                                 # GPU-h held, busy and useful per node and UTC hour, and the holdings that held GPUs
                                                 # idle: `cluster usage` (infra/cluster usage.py) over the console's record on vy-nebius-1
  research pods ssh TARGET [--print] [-- CMD...]  # TARGET = machines.toml name | pod id | pod name; the port is looked up, never hard-coded
  research pods sync TARGET [SRC] [--dest DIR]    # rsync the worktree's git-visible files (default /workspace/src), prune what it shipped before
  research notes checkpoint LANE STATE TEXT...   # CHECKPOINT <sha> (<HH:MM>Z) [open|blocked|final|superseded] line atop the report
  research notes bind LANE --branch B --worktree DIR --pod NAME   # where the lane works (successors naming a predecessor's)
  research notes status [--pods] [--stale-min 30] [--json]  # per lane: freshest of report/checkpoint/commit/uncommitted edit/pod; STALE (exit 1)
  research notes watch [--every 10] [--pods]     # a line each time a lane goes STALE or comes back
  research agents quiet [--hours 2] [--repo DIR] [--branch NAME ...] [--json]   # a coordinator's workers: each worktree (or branch)
                                                 # with no commit, change or live process for --hours is QUIET (exit 1)
  research launch-request <request.json>         # machine side: claim the run dir, start the detached runner (idempotent per id)
  research mem [--ramlock DIR] [--json]           # machine side: cgroup limit, in use, live reservations, available (research.mem)
  research msg send|close|read|names|traffic|ask-daniel|... [--dry-run]
                                                 # messages between agents, on Slack (research/slack.py; skill using-slack)
  research digest --as LANE [--roster FILE] [--watch N TODO | --unwatch N | --thread NAME TS [--channel KEY] | --show]
                                                 # a coordinator's sync on one screen: doorbells, watched threads, landings, the
                                                 # lane's queue changes, its notes with their results so far, TRIGGERs for watched
                                                 # PRs, and Health: the VM, /workspace, the roster's workers and inboxes, PRs off the
                                                 # queue, finished worktrees' .lake, live runs (research/digest.py)

`run` allocates a run id, writes the v0.2 job record, gives the workload RESEARCH_RUN_ID / RESEARCH_RUN_DIR, and hands execution
to the telemetry harness (timeout, sampler, cancel evidence, classifier, tee'd logs).  The workload's cwd is its run dir, locally and
with --on; `--cwd source` (or `source/SUB`, a relative SUB) runs it in the source root (the invoking directory; with --on, the shipped
tree other runs of the commit share), `--cwd clone` (or `clone/SUB`) in a private `git clone --shared` of the source root at its HEAD,
outside the run dir and removed when the workload exits (research/clone.py), an absolute --cwd as given; job.json run.cwd /
run.cwd_basis record it.  It exits with the workload's status; with
--require-result, a zero exit and a missing or invalid result.json exits 65 instead.  Harness flags pass through unchanged
(--tree-sha, --workload-digest, --input name=path, --declared-output, --policy, --no-sampler, ...).  `run` and `fetch --all` refuse
(exit 3) while the runs root's filesystem has less than $RESEARCH_DISK_FLOOR_GB free (default 1.5; 0 turns it off), and name the
eviction that frees it (runs.disk_floor_refusal).
`--on <machine>` ships the tool (and, with --source, a clean git HEAD), stages the request, and has the machine claim the run id
and start the same runner detached; see remote.py.  A machine must be listed in ~/.research/machines.toml for the project.
`--tool` loads a store Tool declaration (research/store/tools_registry.py or dotted.module:NAME): before launch the runner records
conditions, the closure manifest, content-resolved key params, the derivation id and reuse_key into job.json (`store` section);
after the workload exits (any status) it publishes the attempt to the local store (research/store/README.md §4, §10).  `--scratch NAME`
gives the workload a cache dir keyed by reuse_key.  `research data ...` is the store's own surface (research.store.cli / cli_prov).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import checkview, clone
from . import compare as C
from . import record, remote, result
from . import runs as R
from . import waitcap
from .timefmt import both

EX_DATAERR = 65


def _harness_parser(prog: str) -> argparse.ArgumentParser:
    from .telemetry.run import add_args

    ap = argparse.ArgumentParser(prog=prog, add_help=False, allow_abbrev=False)
    add_args(ap)
    return ap


def build_job_from(a: argparse.Namespace, command: list[str]) -> dict:
    """Mirror of the job the harness would build for this namespace (telemetry.run.run), so that pre-writing job.json changes
    nothing the harness records.  Kept in one place; if the harness's call changes upstream, change this."""
    from .telemetry.job import build_job, input_ref, sampler_instruments, sha256_file

    policy_sha = sha256_file(a.policy)[0] if a.policy and Path(a.policy).exists() else None
    policy_version = a.policy_version
    if a.policy and Path(a.policy).exists() and not policy_version:
        try:
            import tomllib

            policy_version = tomllib.loads(Path(a.policy).read_text()).get("policy", {}).get("version")
        except Exception:  # noqa: BLE001
            policy_version = None
    return build_job(project=a.project, task=a.task, job=a.job, attempt=a.attempt, lane=a.lane, generation=a.generation, tree_sha=a.tree_sha,
                     release_tag=a.release_tag, workload_digest=a.workload_digest, stage=a.stage, inputs=[input_ref(x) for x in a.input],
                     profile={k: os.environ.get(k) for k in a.profile_key if k in os.environ}, command=command, declared_outputs=a.declared_output,
                     timeout_s=a.timeout, reservation={"gpu": a.reservation_gpu, "host_memory_bytes": a.reservation_host_memory_bytes},
                     policy_version=policy_version, policy_sha256=policy_sha,
                     instruments=sampler_instruments(a.pss_every) if not a.no_sampler else [], cadence_s=a.cadence, pss_every_s=a.pss_every)


def _parse_expect(items: list[str]) -> dict | None:
    if not items:
        return None
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"research run: --expect wants KEY=VALUE, got {it!r}")
        out[k] = v
    return out


def cmd_run(argv: list[str]) -> int:
    clones: list[Path] = []          # the run's `--cwd clone` home once made; removed however `_cmd_run` ends
    try:
        return _cmd_run(argv, clones)
    finally:
        _drop_clone(clones)


def _cmd_run(argv: list[str], clones: list[Path]) -> int:
    started = time.monotonic()                   # --wait's cap counts the launch too: it bounds the whole tool call
    if "--" not in argv:
        print("research run: the workload command goes after `--`", file=sys.stderr)
        return 2
    i = argv.index("--")
    opts, command = argv[:i], argv[i + 1 :]
    if not command:
        print("research run: empty command after `--`", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="research run", add_help=True, allow_abbrev=False, description=__doc__.split("\n\n")[1])
    ap.add_argument("--runs-dir", default=None, help="registry root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--on", default=None, metavar="MACHINE",
                    help="run on an existing authorized machine listed in the machines file; `auto`: an agent's suite or build, "
                         "queued onto vy-nebius-2 from this checkout, 4 CPUs, 16 GB and 60 min unless the run says otherwise, or "
                         "a slot of a CPU node's research pool when it starts sooner there (research/auto.py's SPILL)")
    ap.add_argument("--project", default=None, help="ownership/resource boundary; required with --on (also recorded by the harness)")
    ap.add_argument("--source", default=None, metavar="GITDIR", help="with --on: ship `git archive HEAD` of this clean checkout; it becomes the source root "
                                                                       "and tree_sha (the workload still runs in its run dir unless --cwd source)")
    ap.add_argument("--send", action="append", default=[], metavar="FILE", help="with --on: ship this file to $RESEARCH_RUN_DIR/inputs/")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="environment for the workload (local or, with --on, on the machine), recorded as "
                                                                                 "the attempt's params.env (values of secret-named keys redacted)")
    ap.add_argument("--exclusive", action="store_true", help="with --on: refuse if another research run is live on the machine")
    ap.add_argument("--wait", action="store_true",
                    help="with --on (or --queue): block until the run ends and exit with its rc (else 1; 3 if refused), for at most "
                         f"{waitcap.WAIT_CAP_S} s with the launch (20 minutes: no tool call blocks longer): then it stops waiting, "
                         f"cancels nothing, names the check-back command (`research status RUN`) and exits {waitcap.STILL_GOING}. "
                         "Without it, run returns once the machine has started the run, so a loop of launches doesn't serialize them")
    ap.add_argument("--wait-every", type=float, default=30.0, metavar="S", help="with --wait: fetch the run's records every S seconds (30)")
    ap.add_argument("--wait-max", type=float, default=None, metavar="S",
                    help=f"with --wait: stop waiting after S seconds instead of {waitcap.WAIT_CAP_S}; never more than {waitcap.WAIT_CAP_S}")
    ap.add_argument("--mem-gb", type=float, default=None, metavar="N",
                    help="with --on: admit only if the machine's memory cgroup has N GB free after live reservations (/workspace/ramlock), "
                         "and reserve them for the run, which the workload reads as $RESEARCH_RUN_MEM_GB (a reservation sets no cgroup "
                         "cap); else MemoryNotAvailable (research.mem). With --queue: the job's memory cap "
                         "(its scope's MemoryMax=), required unless --kind names a registered kind, whose shape gives it")
    ap.add_argument("--mem-wait", type=float, default=0.0, metavar="S",
                    help="with --mem-gb: wait up to S seconds for the memory instead of refusing, in line behind older waiters "
                         "(first come first served, research.mem)")
    ap.add_argument("--preflight", default=None, metavar="TOOL",
                    help="with --on --source: run TOOL's declared preflight (Tool.preflight) on the machine before this run, as a --tool TOOL "
                         "run does, and refuse the run when it fails")
    ap.add_argument("--job-user", default=None, metavar="USER",
                    help="run the workload as USER (an unprivileged system user) in a transient system unit, with no sudo, no home and "
                         "a clean environment; the runner keeps its records as itself (research.jobuser)")
    ap.add_argument("--pool", default=None, metavar="NAME",
                    help="with --on --source: run in the machine's NAME pool (`research`: any lane's research runs on vy-nebius-cpu-1, "
                         "on cores no check slot holds; `proofs`): the run takes a slot there and its cwd is its source, or "
                         "with --cwd clone a private clone of it, for a job that writes into the tree (a Lean check that rewrites a lock); "
                         "its --timeout counts from taking the slot, after a wait of up to --pool-wait for one")
    ap.add_argument("--pool-wait", type=float, default=None, metavar="S",
                    help="with --pool: how long to wait in the pool's line for a slot before exiting 75 unrun (default 4 h)")
    ap.add_argument("--as", dest="lane", default=os.environ.get("RESEARCH_LANE"), metavar="LANE",
                    help="with --pool: the lane the slot names (default $RESEARCH_LANE)")
    q = ap.add_argument_group("the queue (--queue): the one queue over the shared nodes (infra/cluster) places and runs the job; "
                                   "--on then pins its node, and --mem-gb is its memory")
    q.add_argument("--queue", action="store_true", help="hand the job to the queue instead of running it on a named machine (needs --source)")
    q.add_argument("--class", dest="queue_class", default="work", metavar="P", help="its priority class (the description's [priorities]; default work)")
    q.add_argument("--gpus", type=int, default=None, metavar="N",
                   help="with --queue: default its kind's (else 0). With --on MACHINE alone: on a machine whose GPUs gpu-lease hands out, "
                        "the workload waits first come first served for N of them (`gpu-lease N --wait`; the wait counts against "
                        "--timeout); a run without --gpus sees no GPU there")
    q.add_argument("--gpu-model", default=None, metavar="M", help="e.g. rtx-pro-6000; a node with other GPUs is never chosen")
    q.add_argument("--cpus", type=int, default=None, metavar="N", help="default: its kind's (else 1); also the workload's OMP, OpenBLAS, MKL and NumExpr threads unless --env sets them")
    q.add_argument("--max-min", type=int, default=None, metavar="M",
                   help="its longest run in minutes, which the GPU lease enforces (default --timeout, else its kind's max_wall_min); "
                        "without --timeout, a GPU job's runner also allows GPU_WAIT_MIN for the lease to be granted")
    q.add_argument("--kind", dest="queue_kind", default="adhoc", metavar="K",
                   help="what the job is: a kind in infra/cluster/kinds/<owner>.toml supplies its shape, time and question (default adhoc)")
    q.add_argument("--question", default=None, metavar="Q", help="the research question it answers, recorded (default: its kind's)")
    q.add_argument("--disk-gb", type=float, default=None, metavar="N",
                   help="the disk it will write: held on the machine while N more GB would fill the disk past 80%% (released at "
                        "75%%); default its kind's disk_gb, else 20")
    q.add_argument("--phase", dest="queue_phase", choices=("cpu", "gpu"), default=None,
                   help="the phase this run is (default: gpu with --gpus, else cpu); a GPU is booked only for a gpu phase, and only "
                        "around the workload, so the runner's custody runs in the CPU allocation")
    q.add_argument("--preemptible", action="store_true", help="it may be stopped for higher-ranked work and requeued (a guest's must be)")
    q.add_argument("--gpu-host", action="store_true",
                   help="a job without --gpus whose workload takes its GPUs with gpu-lease itself, around each GPU step: its CPU "
                        "allocation runs at a GPU job's weight (node 1: not CPUWeight=idle)")
    q.add_argument("--quiet", action="store_true", help="a timed job: the node is kept quiet while it runs (the node's owner only)")
    q.add_argument("--gpu-ids", default=None, metavar="LIST", help="the GPU indices it must get, comma-separated")
    ap.add_argument("--id", default=None, help="use this run id instead of allocating one")
    ap.add_argument("--claimed", action="store_true", help=argparse.SUPPRESS)          # machine side: the dir was created by launch-request
    ap.add_argument("--cost-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--remote-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--exclusive-lock", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--custody-dir", default=None, help=argparse.SUPPRESS)   # machine side: the minted key + store.toml (store.custody)
    ap.add_argument("--store-read-dir", default=None, help=argparse.SUPPRESS)   # machine side: --store-read's key, deleted when the workload ends
    ap.add_argument("--store-read", action="store_true",
                    help="with --on: the workload gets a read-only key to the store's remote (RESEARCH_STORE_CONFIG and AWS_* in its "
                         "environment, minted here like custody's and deleted when it ends), so its `research data fetch` works on a "
                         "machine whose local store lacks the artifact; needs run custody (the default)")
    ap.add_argument("--custody-r2", dest="custody_r2", action="store_true", default=None,
                    help="with --on (the default there): the machine publishes the attempt and every run file (logs included) to the "
                         "store's remote with a key minted here, so custody does not depend on this machine (store.custody)")
    ap.add_argument("--no-custody-r2", dest="custody_r2", action="store_false",
                    help="with --on: no run custody on the remote; the records stay on the machine until `research fetch --all`")
    ap.add_argument("--custody-ttl", default=None, metavar="TTL", help="with --custody-r2: the minted key's lifetime (default 6h, at most 24h; it must outlast "
                                                                       "the run and its upload: a --timeout over 6h needs it)")
    ap.add_argument("--run-record", action="store_true",
                    help="a local run: the attempt also carries every run file (logs included; what custody keeps) as its run record, as an "
                         "--on run's custody does: for a pod whose run dir nothing else preserves (store.custody)")
    ap.add_argument("--expect", action="append", default=[], metavar="K=V", help="expected workload fingerprint entry (recorded separately from what the workload observes)")
    ap.add_argument("--require-result", action="store_true", help="a zero exit without a valid result.json exits 65")
    ap.add_argument("--stage", default="run", help="stage name recorded in the job (default: run)")
    ap.add_argument("--tool", default=None, metavar="NAME|module:NAME",
                    help="store Tool declaration (research.store): closure manifest, key params, derivation and reuse_key are computed before launch, "
                         "the attempt is published after")
    ap.add_argument("--scratch", action="append", default=[], metavar="NAME",
                    help="tool cache dir under <store>/scratch/<reuse_key>/NAME (cold per commit, on purpose); `triton` exports TRITON_CACHE_DIR, "
                         "any other NAME exports RESEARCH_SCRATCH_<NAME>")
    ap.add_argument("--campaign", default=os.environ.get("RESEARCH_CAMPAIGN"), help="campaign recorded in the attempt's conditions (default $RESEARCH_CAMPAIGN)")
    ap.add_argument("--local-only", default=os.environ.get("RESEARCH_LOCAL_ONLY") or None, metavar="REASON",
                    help="a local run whose store has no usable remote refuses (exit 3): its records would stay on this disk and nothing "
                         "would preserve them. REASON says what does instead; it is printed and recorded in job.json (default "
                         "$RESEARCH_LOCAL_ONLY). A usable remote is pushed to either way; --on MACHINE gets custody on the remote instead")
    ap.add_argument("--source-identity", default=None, help=argparse.SUPPRESS)   # machine side: JSON {commit, tree, dirty} of the shipped source
    ns, harness_opts = ap.parse_known_args(opts)
    ns.started = started
    if ns.wait_max is not None:
        try:
            if not ns.wait:
                raise waitcap.OverCap("--wait-max is for --wait")
            waitcap.max_s(ns.wait_max)
        except waitcap.OverCap as e:
            print(f"research run: {e}", file=sys.stderr)
            return 2
    if ns.on == "auto" and (why := _auto_placement(ns, harness_opts)):
        print(f"research run --on auto: {why}", file=sys.stderr)
        return 2
    if getattr(ns, "auto", False):
        _auto_size(ns, harness_opts, command)
        if (harness_opts := _auto_spill(ns, harness_opts)) is None:
            return 2
    if ns.claimed:
        _survive_stray_signals()
    mem_reservation = os.environ.pop("RESEARCH_MEM_RESERVATION", None) if ns.claimed else None    # research.mem: this runner's, not the workload's
    if ns.run_record and (ns.on or ns.queue):
        print("research run: --run-record is for a local run; --on MACHINE and --queue keep every run file through their custody "
              "(--custody-r2, the default there)", file=sys.stderr)
        return 2
    if (ns.on or ns.queue) and (why := remote.pss_bound_problem(harness_opts, Path(ns.source).expanduser() if ns.source else None)):
        print(f"research run: refused: {why}", file=sys.stderr)
        return 3
    if ns.pool:
        from .pods.nebius import slot_only  # noqa: PLC0415

        cwd = remote.flag_value(harness_opts, "--cwd")
        why = ("is for --on, not --queue" if ns.queue
               else "needs --on MACHINE and --source (slot.py runs from the shipped tree)" if not (ns.on and ns.source)
               else "needs a lane: --as LANE or $RESEARCH_LANE" if not ns.lane
               else f"runs from its source or a clone of it (slot.py's path is the tree's), not --cwd {cwd}"
               if cwd not in (None, "source", "clone") else None)
        if why:
            print(f"research run: --pool {why}", file=sys.stderr)
            return 2
        try:
            timeout = remote.timeout_seconds(remote.flag_value(harness_opts, "--timeout"))
            wait_s = ns.pool_wait if ns.pool_wait is not None else slot_only.POOL_WAIT_S
            command = slot_only.in_pool(ns.pool, ns.lane, command, timeout, wait_s)
        except ValueError as e:
            print(f"research run: --pool: {e}", file=sys.stderr)
            return 2
        if timeout:
            harness_opts = [*harness_opts, "--timeout", f"{wait_s + timeout:g}"]
            print(f"research run: --pool {ns.pool}: --timeout {timeout:g} s counts from taking a slot (slot.py --run-max); it waits "
                  f"up to {wait_s / 60:g} min for one (--pool-wait), then exits 75 without running", file=sys.stderr)
        if ns.mem_gb is not None:
            # the slot admits and reserves the run, not a launch-time reservation, which was held through the wait for a slot
            # beside the slot's once it ran (INF-61, cpu-3 18 Oct: two waiting runs held 128 GB, admission sat at -89 GB); the
            # slot takes the larger of its mem= and this (24 Oct: a 450 GB run in proofs' 193 GB slot p5 reserved 193)
            ns.env = [*ns.env, f"CHECK_RUN_MEM_GB={ns.mem_gb:g}"]
            print(f"research run: --pool {ns.pool}: --mem-gb {ns.mem_gb:g} goes to the slot, which waits for and reserves the larger of "
                  "its own mem= and that", file=sys.stderr)
            ns.mem_gb = None
        if cwd is None:
            harness_opts = [*harness_opts, "--cwd", "source"]
    if ns.queue:
        try:
            harness_opts = _normalize_timeout(harness_opts)
        except ValueError as e:
            print(f"research run: --timeout: {e}", file=sys.stderr)
            return 2
        return _run_queued(ns, harness_opts, command)
    queue_only = [flag for flag, given in (("--gpus", ns.gpus is not None and not ns.on), ("--cpus", ns.cpus is not None),
                                           ("--gpu-model", ns.gpu_model), ("--max-min", ns.max_min is not None),
                                           ("--kind", ns.queue_kind != "adhoc"), ("--question", ns.question),
                                           ("--phase", ns.queue_phase), ("--class", ns.queue_class != "work"),
                                           ("--preemptible", ns.preemptible), ("--quiet", ns.quiet), ("--gpu-ids", ns.gpu_ids),
                                           ("--gpu-host", ns.gpu_host),
                                           ("--disk-gb", ns.disk_gb is not None))
                  if given]
    if queue_only:
        print(f"research run: {', '.join(queue_only)} declare a job to the queue and mean nothing without --queue", file=sys.stderr)
        return 2
    if ns.mem_gb is not None and not ns.on:
        print("research run: --mem-gb admits a run on a shared machine (--on MACHINE); a local run has no reservation to take", file=sys.stderr)
        return 2
    if ns.preflight and not (ns.on and ns.source):
        print("research run: --preflight runs a tool's preflight on the machine, from the tree --source ships (--on MACHINE --source GITDIR)",
              file=sys.stderr)
        return 2
    if ns.store_read and not ns.on:
        print("research run: --store-read hands a machine's workload a read-only store key (--on MACHINE); here the store's own "
              "config already applies", file=sys.stderr)
        return 2
    if (ns.custody_r2 or ns.custody_ttl) and not ns.on:
        print("research run: --custody-r2 / --custody-ttl are for --on MACHINE (a local run's records are already on this disk: "
              "`research data custody RUN --publish` puts them on the remote)", file=sys.stderr)
        return 2
    if ns.custody_r2 is False and ns.custody_ttl:
        print("research run: --custody-ttl is the lifetime of the --custody-r2 key; --no-custody-r2 mints none", file=sys.stderr)
        return 2
    try:
        harness_opts = _normalize_timeout(harness_opts)
    except ValueError as e:
        print(f"research run: --timeout: {e}", file=sys.stderr)
        return 2
    if ns.on and ns.custody_r2 is not False:
        why = _custody_ttl_problem(ns.custody_ttl, harness_opts)
        if why:
            print(f"research run: {why}", file=sys.stderr)
            return 2
    if ns.project:
        harness_opts = ["--project", ns.project, *harness_opts]
    root = R.runs_root(ns.runs_dir)
    if not ns.claimed and (why := R.disk_floor_refusal(root)):
        print(f"research run: refused: {why}", file=sys.stderr)
        return 3
    run_id = ns.id or R.new_run_id()
    run_dir = root / run_id
    if ns.on:
        return _run_remote(ns, run_id, run_dir, harness_opts, command)
    job_pw = job_dir = None
    if ns.job_user:
        from . import jobuser

        try:
            job_pw = jobuser.check_user(ns.job_user)
            job_dir = jobuser.check_run_dir(run_dir, run_id)
        except jobuser.Refused as e:
            print(f"research run: refused: {e}", file=sys.stderr)
            return 3
    # a claimed run's custody was settled by its launcher (--custody-r2, or --no-custody-r2 and `research fetch --all`)
    no_remote = None if ns.claimed else _no_usable_remote()
    if no_remote and not ns.local_only:
        print(f"research run: refused: this store has no usable remote ({no_remote}), so the run's records would stay on this disk "
              f"and nothing would preserve them. Run it with --on MACHINE (custody on the store's remote), give this store a remote "
              f"(~/.research/store.toml [remote], or R2_ENDPOINT, R2_BUCKET and AWS_* in the environment), or name what preserves "
              f"it with --local-only REASON ($RESEARCH_LOCAL_ONLY)", file=sys.stderr)
        return 3
    if no_remote:
        print(f"research run: no usable remote ({no_remote}); --local-only: {ns.local_only}", file=sys.stderr)
    env_map = _parse_env(ns.env)
    if env_map is None:
        return 2
    if ns.claimed and not env_map:
        env_map = _request_env(ns.remote_json)
    if run_dir.exists() and ((run_dir / "job.json").exists() or not ns.claimed):
        print(f"research run: {run_dir} already exists; a rerun gets a fresh id", file=sys.stderr)
        return 2
    hp = _harness_parser("research run")
    try:
        a = hp.parse_args(["--attempt-dir", str(run_dir), "--stage", ns.stage, "--job", run_id, *harness_opts])
    except SystemExit:
        return 2
    if ns.claimed and not os.environ.get("RESEARCH_STORE"):
        os.environ["RESEARCH_STORE"] = str(root.parent / "store")     # machine side: the store lives beside the runs, <root>/store (what `data pull` reads)
    cwd_basis = _resolve_cwd(a)
    remote_info = json.loads(ns.remote_json) if ns.remote_json else None
    clone_rec = None
    clone_env: dict[str, str] = {}
    if cwd_basis == "clone":
        try:
            clone_rec, clone_env = _make_clone(a, ns, root, run_id, remote_info, clones)
        except clone.Refused as e:
            return _refuse(ns, run_dir, run_id, f"--cwd {a.cwd}: {e}", "CloneRefused", mem_reservation)
    try:
        prep = _prepare_store(ns, a, command, run_id, run_dir, env_map)
    except (ImportError, LookupError, TypeError) as e:
        print(f"research run: --cwd {a.cwd} / --tool {ns.tool}: {e}", file=sys.stderr)
        return 2
    if no_remote:
        prep["section"]["local_only"] = {"reason": ns.local_only, "no_remote": no_remote}
    if a.cwd is None:
        # the workload runs in its own run dir, locally and on a machine (a relative `> result.json` used to land in the checkout, and
        # on a machine in <root>/src/<sha>/, which every run of that commit shares: concurrent runs raced on it)
        a.cwd = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=ns.claimed)
    _place_dirty_capture(prep, run_dir)
    cost = json.loads(ns.cost_json) if ns.cost_json else None
    if ns.claimed and remote_info is not None:
        try:
            remote_info["tool_running"] = remote.check_tool(remote_info.get("tool_identity"))
        except remote.ResearchVersionMismatch as e:
            return _refuse(ns, run_dir, run_id, str(e), type(e).__name__, mem_reservation)
    job = record.extend_job(build_job_from(a, command), run_id=run_id, runs_root=root, expected_fingerprint=_parse_expect(ns.expect),
                            cost=cost, remote=remote_info, store=prep["section"], cwd={"path": a.cwd, "basis": cwd_basis})
    if ns.job_user:
        job["run"]["job_user"] = ns.job_user
    if clone_rec:
        job["run"]["cwd_clone"] = clone_rec
    try:                                                          # which build each installed binary is (binprov.py); never stops a run
        from .binprov import snapshot

        if bins := snapshot():
            job["run"]["binaries"] = bins
    except Exception as e:  # noqa: BLE001
        job["run"]["binaries_error"] = f"{type(e).__name__}: {e}"
    from .telemetry.job import write_json_atomic
    from .telemetry.run import run as harness_run

    write_json_atomic(run_dir / "job.json", job)
    if ns.claimed and remote_info and remote_info.get("request"):
        remote.ack_launch(Path(remote_info["request"]).parent / remote.LAUNCH_ACK, remote.FIRST_RECORD)
    os.environ.update(result.env_for(run_id, run_dir, source_sha=getattr(a, "tree_sha", None)))
    os.environ.update(prep["env"])
    # --env is the workload's: set for the harness (which spawns it with this environment), then restored so the store publish below
    # runs with the caller's environment (a workload-only AWS_* must not become the push credentials)
    apply_env = {k: v for k, v in env_map.items() if not (ns.claimed and k == "PYTHONPATH")}   # launch_request composed it: tool, tree roots, then this
    if ns.claimed and (remote_info or {}).get("gpus"):
        apply_env.pop("CUDA_VISIBLE_DEVICES", None)          # launch_request hid the GPUs (or gpu-lease names them): --env can't undo it
    apply_env = {**clone_env, **apply_env}
    saved_env = {k: os.environ.get(k) for k in apply_env}
    os.environ.update(apply_env)
    spawn, mode, unsafe = command, None, []
    if ns.job_user:
        # inside a queued GPU phase's lease (the request's workload wrap), so gpu-lease and its bookkeeping stay the runner's
        wrap = list(_request(ns.remote_json).get(remote.QUEUE_WORKLOAD_WRAP) or []) if ns.claimed else []
        n = len(wrap) if wrap and command[: len(wrap)] == wrap else 0
        spawn = [*command[:n], *jobuser.spawn_argv(command[n:], ns.job_user, jobuser.unit_name(run_id), list({**clone_env, **env_map}))]
    print(f"research: run {run_id} -> {run_dir}" + (f"  cwd={a.cwd}" if a.cwd else ""), file=sys.stderr)
    try:
        if ns.job_user:
            mode = jobuser.prepare(job_dir)
        rc = harness_run(a, spawn)
    finally:
        if mode is not None:            # before anything reads the run dir
            removed, unsafe = jobuser.finish(job_dir, jobuser.unit_name(run_id), ns.job_user, job_pw.pw_uid, mode)
            if removed:
                print(f"research: --job-user: deleted {len(removed)} entr{'y' if len(removed) == 1 else 'ies'} the job left that "
                      f"{'is' if len(removed) == 1 else 'are'} neither a file nor a directory: {', '.join(removed[:10])}", file=sys.stderr)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if ns.exclusive_lock:
            remote.release_exclusive(Path(ns.exclusive_lock).parent, run_id)
        if mem_reservation:
            from .mem import release

            release(mem_reservation, run_id)
        _drop_clone(clones, run_dir)
    if unsafe:
        print(f"research: --job-user: refused to publish: {len(unsafe)} entr{'y' if len(unsafe) == 1 else 'ies'} in {run_dir} still "
              f"belong to {ns.job_user} ({', '.join(unsafe[:10])}), so nothing here reads the run dir; its records "
              f"stay on this machine", file=sys.stderr)
        return rc or 3
    if ns.store_read_dir:               # the workload is done with its read-only key
        from .store.custody import CRED_FILE

        (Path(ns.store_read_dir) / CRED_FILE).unlink(missing_ok=True)
    if not ns.custody_dir:
        _publish_attempt(run_dir, prep, run_record=ns.run_record)
    s = R.summarize(run_dir)
    line = f"research: run {run_id} {s['state']} rc={s['rc']} class={s['class'] or '-'} result={s['result']}"
    if s["result"] == "absent":
        line += f" (looked for {run_dir / 'result.json'}: write $RESEARCH_RUN_DIR/result.json)"
    if s["validation"]:
        line += f" validation={s['validation']}"
    print(line + f"  (research inspect {run_id})", file=sys.stderr)
    out = rc
    if ns.require_result and rc == 0 and not s["result_valid"]:
        print(f"research: --require-result: the workload exited 0 but result.json is {s['result']}; exiting {EX_DATAERR}", file=sys.stderr)
        out = EX_DATAERR
    if ns.custody_dir:                  # last: once .custody is written nothing more goes into the run dir (launcher.log included)
        from .store import custody as CU

        conds = prep.get("conditions")
        if conds is not None:
            from . import env as E

            conds["exec.load_end"] = E.load_now()
        print(f"research: custody: publishing the attempt and every run file to the store's remote; the verdict goes to {run_dir / CU.MARKER} "
              f"(`research data custody {run_id}` from any machine)", file=sys.stderr)
        CU.runner_publish(run_dir, Path(ns.custody_dir), tool=prep.get("tool"), conditions=conds)
    return out


#: The longest custody key ``research run`` mints on request; the default stays 6 h.
CUSTODY_TTL_CAP_S = 24 * 3600


def _harness_timeout(harness_opts: list[str]) -> float | None:
    """The harness ``--timeout`` (seconds) among the pass-through options, or ``None`` (normalized by :func:`_normalize_timeout`)."""
    v = remote.flag_value(harness_opts, "--timeout")
    return None if v is None else float(v)


def _normalize_timeout(harness_opts: list[str]) -> list[str]:
    """``--timeout 40m`` -> ``--timeout 2400``: the harness, the custody-key guard and the lease extension all read seconds.
    ValueError on a malformed value, so it is refused here and not by the machine after launch."""
    out, i = [], 0
    while i < len(harness_opts):
        o = harness_opts[i]
        if o == "--timeout" and i + 1 < len(harness_opts):
            out += [o, f"{remote.timeout_seconds(harness_opts[i + 1]):g}"]
            i += 2
            continue
        out.append(f"--timeout={remote.timeout_seconds(o.split('=', 1)[1]):g}" if o.startswith("--timeout=") else o)
        i += 1
    return out


def _custody_ttl_problem(custody_ttl: str | None, harness_opts: list[str]) -> str | None:
    """Why a custody launch is refused: an explicit ``--custody-ttl`` over :data:`CUSTODY_TTL_CAP_S`, or a ``--timeout`` longer than
    the default key's lifetime without an explicit ``--custody-ttl`` (the key would expire before the run's upload)."""
    from research.store import credentials as C  # noqa: PLC0415
    from research.store.custody import DEFAULT_TTL  # noqa: PLC0415

    try:
        ttl = C.parse_ttl(custody_ttl or DEFAULT_TTL)
    except C.MintError as e:
        return str(e)
    if custody_ttl and ttl > CUSTODY_TTL_CAP_S:
        return f"--custody-ttl {custody_ttl} is over the {CUSTODY_TTL_CAP_S // 3600} h cap"
    timeout = _harness_timeout(harness_opts)
    if not custody_ttl and timeout is not None and timeout > ttl:
        return (f"--timeout {timeout:g} s outlives the custody key's default lifetime ({DEFAULT_TTL}): the upload would fail; "
                f"pass --custody-ttl (at most {CUSTODY_TTL_CAP_S // 3600}h) longer than the run")
    return None


def _survive_stray_signals() -> None:
    """Machine side: the runner's argv carries the workload's command, so `pkill -f <pattern>` aimed at the workload also hits the runner,
    which then died without recording or publishing the run.  SIGTERM and SIGHUP are logged and otherwise ignored: the runner keeps
    supervising and publishes when the workload exits.  A run is stopped through its workload's process group (status.json `pgid`);
    SIGKILL still ends the runner.  A Python handler, not SIG_IGN, so the workload it spawns gets the default disposition back at exec."""
    import signal
    import time

    def note(signum: int, _frame: object) -> None:
        print(f"research: runner got {signal.Signals(signum).name} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, ignored: it records "
              f"and publishes the run when the workload exits; stop the workload with `research cancel RUN` from any VM, or "
              f"`kill -TERM -<pgid>` (status.json) here, and never SIGKILL the runner", file=sys.stderr, flush=True)

    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, note)


CWD_SOURCE = "source"


def _resolve_cwd(a: argparse.Namespace) -> str:
    """The workload's cwd, resolved in place on `a.cwd`; returns its basis for the record.

    The SOURCE ROOT is this process's cwd: the invoking directory locally, the shipped tree <root>/src/<sha>/ on a machine (launch_request
    starts the runner there; the store's closure and tool declarations are read from it).  `--cwd source` names it, `--cwd source/SUB` or
    a relative `--cwd SUB` a directory under it (an explicit opt-in to running inside a tree other runs may share); an absolute `--cwd`
    is taken as given.  `--cwd clone` (or `clone/SUB`): basis `clone`, `a.cwd` left as given for `_make_clone`, which puts the run's
    own clone of the source root in its place.  No `--cwd`: basis `run_dir`, set by the caller once the store preparation has read
    the source root."""
    if a.cwd is None:
        return "run_dir"
    raw = a.cwd
    if os.path.isabs(raw):
        return "explicit"
    if clone.sub(raw) is not None:
        return "clone"
    if raw == CWD_SOURCE or raw.startswith(CWD_SOURCE + "/"):
        raw = raw[len(CWD_SOURCE):].lstrip("/")
    a.cwd = os.path.abspath(os.path.join(os.getcwd(), raw))
    return "source"


def _make_clone(a: argparse.Namespace, ns: argparse.Namespace, runs_root: Path, run_id: str, remote_info: dict | None,
                clones: list[Path]) -> tuple[dict, dict[str, str]]:
    """`--cwd clone[/SUB]`: clone the source root (this process's cwd, as for `--cwd source`) for this run alone (research.clone) and
    point `a.cwd` at it, or at SUB in it.  On a machine the clone must be the shipped commit; locally the checkout must be clean.
    The directory to remove once the run is over goes on `clones` as soon as it exists.  Returns job.json's run.cwd_clone and the
    workload's environment (`_clone_imports`); clone.Refused when it cannot."""
    sub = clone.sub(a.cwd) or ""
    source_root = Path(os.getcwd())
    shipped = (remote_info or {}).get("source_sha") if ns.claimed else None
    if ns.claimed and not shipped:
        raise clone.Refused("this run has no shipped source tree to clone (--source GITDIR ships one)")
    home = clone.home(runs_root, run_id, machine=ns.claimed)
    clones.append(home)
    tree = home / "tree"
    commit = clone.make(source_root, tree, commit=shipped, clean=not ns.claimed)
    try:
        inside = (tree / sub).resolve().is_relative_to(tree.resolve()) and (tree / sub).is_dir()
    except (OSError, RuntimeError):
        inside = False
    if not inside:
        raise clone.Refused(f"{sub} is not a directory of commit {commit}")
    a.cwd = str(tree / sub) if sub else str(tree)
    env = _clone_imports(source_root, tree, home, remote_info) if ns.claimed else {}
    return {"root": str(tree), "of": str(source_root), "commit": commit}, env


def _clone_imports(of: Path, tree: Path, home: Path, remote_info: dict | None) -> dict[str, str]:
    """A machine's runner has the shipped tree's import roots on PYTHONPATH (launch_request), so the workload would import its code
    from the shared tree: a write beside a module's __file__ would land there, and the run would execute that tree's files, whatever
    another run left in them, while the record describes the clean clone.  Each entry under `of` is moved to the same directory in
    the clone, and `of`'s shim (`pythonpath.shim_packages`: <root>/cache/pyshim/<key>, outside `of` but linking into it, so
    `import verity` reached the shared tree) is replaced by a shim of the clone under `home`, removed with it; in place,
    run.remote.pythonpath too.  Bytecode goes beside the clone: under <root>/cache/pycache every clone path would add its own
    copies, never removed.  Locally the workload imports from whatever its interpreter has installed (an editable install points
    at the invoking checkout); nothing is moved."""
    from .pythonpath import shim_dir, shim_packages

    base = os.path.realpath(of)

    def move(p: str) -> str:
        real = os.path.realpath(p)
        if real == base or real.startswith(base + os.sep):
            return str(tree / os.path.relpath(real, base))
        return shim_dir(tree, pkgs, home / "pyshim") if (pkgs := shim_packages(p, base)) else p

    if remote_info is not None and remote_info.get("pythonpath"):
        remote_info["pythonpath"] = [move(p) for p in remote_info["pythonpath"]]
    entries = [move(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return {"PYTHONPATH": os.pathsep.join(entries), "PYTHONPYCACHEPREFIX": str(home / "pycache")}


#: how many times `_end_group` looks for the group's members after its SIGKILL, GROUP_END_POLL_S apart, before removing the clone anyway
GROUP_END_POLLS = 100
GROUP_END_POLL_S = 0.05


def _drop_clone(clones: list[Path], run_dir: Path | None = None) -> None:
    """Remove the run's clone, if `clones` still holds it, once its workload has exited or the run ended before it; a failure is
    reported, never fatal.  `run_dir`: the workload ran, and whatever is left of its process group (status.json pgid) is ended
    first (`_end_group`)."""
    while clones:
        home = clones.pop()
        if run_dir is not None:
            _end_group(run_dir)
        if why := clone.remove(home):
            print(f"research: --cwd clone: could not remove {home} ({why}); remove it by hand", file=sys.stderr)


def _end_group(run_dir: Path) -> None:
    """SIGKILL what is left of the workload's process group once its leader has exited: a member that ignored the SIGTERM `research
    cancel` sends, or that the workload left running, would keep writing into the clone while it is removed (rmtree then fails with
    ENOTEMPTY and the clone stays).  The clone is the run's own, so nothing else needs those processes."""
    import signal
    import time

    from .telemetry.procs import live_group

    try:
        pgid = int(json.loads((run_dir / "status.json").read_text())["pgid"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if not (left := live_group(pgid)):
        return
    print(f"research: --cwd clone: {len(left)} process(es) of the workload's group {pgid} outlived it "
          f"({', '.join(str(st['pid']) + ' ' + st['comm'] for st in left[:5])}); SIGKILL to the group before removing the clone",
          file=sys.stderr)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    for _ in range(GROUP_END_POLLS):
        if not live_group(pgid):
            return
        time.sleep(GROUP_END_POLL_S)


def _refuse(ns: argparse.Namespace, run_dir: Path, run_id: str, reason: str, fault: str, mem_reservation: str | None) -> int:
    """Refuse the run before its workload (exit 3).  A claimed run records refused.json, deletes its custody key (nothing will
    publish, so it goes now and not at its expiry, as launch_request's refusals do) and gives back its exclusive lock and memory
    reservation; a local one has nothing to record yet."""
    print(f"research run: refused: {reason}", file=sys.stderr)
    if not ns.claimed:
        return 3
    from .store.custody import CRED_FILE
    from .telemetry.events import utc_now_ms
    from .telemetry.job import write_json_atomic

    write_json_atomic(run_dir / "refused.json", {"run_id": run_id, "reason": reason, "fault": fault, "at_utc": utc_now_ms()})
    if ns.custody_dir:
        (Path(ns.custody_dir) / CRED_FILE).unlink(missing_ok=True)
    if ns.store_read_dir:
        (Path(ns.store_read_dir) / CRED_FILE).unlink(missing_ok=True)
    if ns.exclusive_lock:
        remote.release_exclusive(Path(ns.exclusive_lock).parent, run_id)
    if mem_reservation:
        from .mem import release

        release(mem_reservation, run_id)
    return 3


def _parse_env(items: list[str]) -> dict[str, str] | None:
    """--env K=V ... -> {K: V}; None (after a message) on a malformed item."""
    env: dict[str, str] = {}
    for kv in items:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            print(f"research run: --env wants K=V, got {kv!r}", file=sys.stderr)
            return None
        env[k] = v
    return env


def _request(remote_json: str | None) -> dict:
    """Machine side: the launch request (request.json) that --remote-json names; {} when unreadable."""
    try:
        req = json.loads(Path(json.loads(remote_json or "{}")["request"]).read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return req if isinstance(req, dict) else {}


def _request_env(remote_json: str | None) -> dict[str, str]:
    """Machine side: the `env` launch-request applied to this process, read back from the request.json that --remote-json names, so the
    attempt records it as params.env like a local --env (the values never travel on a command line); {} when unreadable."""
    try:
        return {str(k): str(v) for k, v in (_request(remote_json).get("env") or {}).items()}
    except AttributeError:
        return {}


def unredact_key_params(command: list[str], redacted: list[str], key_params: set[str]) -> list[str]:
    """`redact_argv` judges a flag by its name (`--auth` looks like a credential), but a flag the Tool declares as a key param is
    part of the derivation identity: a scientific choice (`--auth included-hash`), never a secret.  Put those values back, token
    for token (`redact_argv` keeps the argv's length), unless the value itself is secret-shaped."""
    from .telemetry.job import SECRET_VALUE

    def dest(flag: str) -> str:
        return flag.lstrip("-").replace("-", "_")

    out = list(redacted)
    for i, (orig, red) in enumerate(zip(command, redacted)):
        if orig == red:
            continue
        if orig.startswith("--") and "=" in orig:
            flag, _, value = orig.partition("=")
            if dest(flag) in key_params and not SECRET_VALUE.match(value):
                out[i] = orig
        elif i and command[i - 1].startswith("--") and dest(command[i - 1]) in key_params and not SECRET_VALUE.match(orig):
            out[i] = orig
    return out


def _prepare_store(ns: argparse.Namespace, a: argparse.Namespace, command: list[str], run_id: str, run_dir: Path, env_map: dict[str, str] | None = None) -> dict:
    """Before launch: load the Tool (errors propagate: the user asked for it), collect conditions, resolve params and inputs to
    content identity, compute the closure manifest, derivation id and reuse_key, create --scratch dirs.  Everything after the
    Tool load is best effort: a failure is recorded in the `store` section, never a reason not to run.  `env_map` (--env) is
    recorded as `params.env`, values redacted where the NAME matches SECRET_KEY; it is not part of the derivation (a Tool that
    depends on an environment variable declares it as a key param)."""
    import tempfile

    from . import env as E
    from .store import attempt as A
    from .store import derivation as D
    from .store.tool import content_identity
    from .telemetry.job import SECRET_KEY, redact_argv

    source_root = Path(a.cwd or os.getcwd())
    tool = None
    if ns.tool:
        from .store import tools_registry

        tool = tools_registry.load(ns.tool, source_root)
    argv_red, _ = redact_argv(command)
    env_out: dict[str, str] = {}
    if ns.campaign:
        env_out["RESEARCH_CAMPAIGN"] = ns.campaign
    conds = None
    capture = Path(tempfile.mkdtemp(prefix=f"research-{run_id}-")) / E.DIRTY_CAPTURE   # the run dir is made after this returns
    try:
        hint = json.loads(ns.source_identity) if ns.source_identity else None
        shipped = getattr(a, "tree_sha", None)
        if hint and shipped and hint.get("commit") != shipped:   # a launcher that read its checkout before shipping: only the sha is known
            hint = {"commit": shipped}
        conds = E.conditions(run_dir=run_dir, project=ns.project, campaign=ns.campaign, exclusive=bool(ns.exclusive_lock), source_root=source_root,
                             source_hint=hint, dirty_capture=capture)
        if tool is not None:
            conds.update(tool.tool_conditions(conds))
            params, key = tool.resolve_params(command, source_root)
            public = set(tool.key_params)
            params = {k: ("<redacted>" if k not in public and SECRET_KEY.search(k) and isinstance(v, str) else v) for k, v in params.items()}
            params["argv"] = unredact_key_params(command, argv_red, public)
            closure = tool.closure_manifest(source_root)
            ident = {"name": tool.name, "version": tool.version, "spec": tool.spec}
            key_params, key_conditions = list(tool.key_params), list(tool.key_conditions)
        else:
            # undeclared: the whole (redacted) command line is the request, and nothing is reusable (reuse_key None below):
            # argv cannot name the files, environment or data the command reads
            params, key, closure = {"argv": argv_red}, {"argv": argv_red}, {}
            ident = {"name": Path(command[0]).name, "version": None, "spec": None}
            key_params, key_conditions = ["argv"], []
        if env_map:
            params["env"] = {k: ("<redacted>" if SECRET_KEY.search(k) else v) for k, v in env_map.items()}
        inputs = {}
        for spec in a.input:                         # harness --input name=path: a fetched artifact is an input; anything else enters the derivation
            name, _, path = spec.partition("=") if "=" in spec else (Path(spec).name, "=", spec)
            if path.startswith("art:"):              # a store id names the input directly (no fetched copy needed on this machine)
                inputs[name] = path
                continue
            cid = content_identity(path, source_root)   # by content, as the key param `input.<name>` (inputs hold art: ids only)
            if isinstance(cid, str) and cid.startswith("art:"):
                inputs[name] = cid
            else:
                key[f"input.{name}"] = cid
        drv = D.derivation_id(ident, inputs, key, conds, key_conditions)
        rk = D.reuse_key(drv, conds) if tool is not None else None
        sroot = A.store_root()
        scratch = {}
        for name in ns.scratch:
            p = D.scratch_dir(sroot, rk, run_id, name)
            scratch[name] = str(p)
            env_out.update(D.scratch_env(name, p))
        section = {"tool": ident, "derivation": drv, "reuse_key": rk, "key_params": key_params, "key_conditions": key_conditions, "params": params,
                   "key_params_resolved": key, "inputs": inputs, "closure": closure, "conditions": conds, "scratch": scratch, "campaign": ns.campaign,
                   "store_root": str(sroot)}
    except Exception as e:  # noqa: BLE001
        section = {"error": f"{type(e).__name__}: {e}", "tool": {"name": tool.name, "version": tool.version, "spec": tool.spec} if tool else None,
                   "campaign": ns.campaign}
        print(f"research run: store preparation failed ({section['error']}); running anyway, the attempt will be published from the records", file=sys.stderr)
    return {"tool": tool, "conditions": conds, "section": section, "env": env_out, "dirty_capture": capture}


def _place_dirty_capture(prep: dict, run_dir: Path) -> None:
    """Move the dirty source's capture (env.DIRTY_CAPTURE) into the run dir, and name it in the store section with the digest it
    reproduces; custody uploads it with the run's own files, and `evict_src` lets the tree go only once that record is preserved.
    No capture (a clean source, `too-large`, any error) leaves the section without one, and the tree stays the run's source."""
    import hashlib
    import shutil

    cap: Path = prep["dirty_capture"]
    try:
        conds = prep["section"].get("conditions") or {}
        if cap.is_file() and conds.get("source.dirty"):
            dest = run_dir / cap.name
            shutil.move(cap, dest)
            with open(dest, "rb") as f:
                sha = hashlib.file_digest(f, "sha256").hexdigest()
            prep["section"]["dirty_capture"] = {"path": cap.name, "bytes": dest.stat().st_size, "sha256": sha,
                                                "dirty_digest": conds.get("source.dirty_digest")}
    except OSError as e:
        prep["section"].pop("dirty_capture", None)
        print(f"research run: the dirty source was not captured ({type(e).__name__}: {e}); its tree stays its only copy", file=sys.stderr)
    finally:
        shutil.rmtree(cap.parent, ignore_errors=True)


def _no_usable_remote() -> str | None:
    """Why the store a local run publishes into (`_publish_attempt`) has no remote it could push to, or None when it has one:
    no `[remote]` and no R2 environment, or a configured remote that does not load here (credentials unset). A failure to tell
    is a reason too: this guard refuses rather than lets an unpreserved run through."""
    from .secret_hint import lost
    from .store import remote as RM

    try:
        if RM.load_remote() is not None:
            return None
        p = RM.config_path()
        if p.is_file():
            return f"{p} has no [remote] table"
        if "RESEARCH_STORE_CONFIG" in os.environ:
            return f"$RESEARCH_STORE_CONFIG names {p}, which does not exist"
        return lost("R2_ENDPOINT", "R2_BUCKET") or f"no {p}, and R2_ENDPOINT / R2_BUCKET are unset"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _publish_attempt(run_dir: Path, prep: dict, *, run_record: bool = False) -> None:
    from . import env as E
    from .store import attempt as A

    conds = prep.get("conditions")
    if conds is not None:
        conds["exec.load_end"] = E.load_now()
    out = A.publish(run_dir, tool=prep.get("tool"), conditions=conds, run_record=run_record)
    if out.get("published"):
        if out.get("already"):
            pres = "  (already in the store)"
        elif out.get("preserved"):
            pres = "  PRESERVED on the remote"
        elif out.get("preserve_error"):
            pres = f"  NOT preserved: {out['preserve_error']}  (`research data push {run_dir.name}` retries)"
        else:
            pres = f"  local only ({out.get('preserve_skipped', 'no remote')}; `research data push {run_dir.name}` with credentials preserves it)"
        print(f"research: attempt published: {out['n_outputs']} outputs drv={out['derivation']} reuse={out['reuse_key'] or '-'}{pres}", file=sys.stderr)
    else:
        print(f"research: attempt not published: {out.get('error')}  (store_publish_error.json in the run dir; `research data attempt publish` retries)",
              file=sys.stderr)


GPU_WAIT_MIN = 120
# a queued job's thread pools take its --cpus, as a pod's take its quota (pod_bootstrap.sh): left alone, BLAS and OpenMP start one
# thread per host core inside a scope of a few CPUs. The job's own --env still decides.
THREAD_POOL_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


#: `--on auto`: an agent's suite, build or probe belongs on a node, not its VM (every handle, at Daniel's request, 18 Oct). It is
#: queued onto vy-nebius-2's Verity share (infra/cluster/descriptions/nebius.toml: whole cores of its own inside its check slots,
#: never a slot a train waits for; it waits on the node while none are free) as kind AUTO_KIND (infra/cluster/kinds/infra.toml),
#: or spills to a CPU node's pool where it starts sooner (research.auto). Not node 1, whose CPUs trains and lanes' direct runs
#: already oversubscribe (top, 02:53Z).
AUTO_KIND = "agent-test"
#: AUTO_KIND's shape and time where the run gives none (cluster.kinds' cpu-test phase, kinds/infra.toml's max_wall_min): what a
#: pool slot must give for the run to spill there
AUTO_CPUS, AUTO_MEM_GB, AUTO_MAX_MIN = 4, 16, 60


def _auto_placement(ns: argparse.Namespace, harness_opts: list[str]) -> str | None:
    """`--on auto` is `--queue --on AUTO_NODE --kind AUTO_KIND --cwd source` from this checkout (its --source) as project
    AUTO_PROJECT: the kind gives what the run doesn't declare (4 CPUs, 16 GiB, 60 min; a suites.py run of named suites,
    `_auto_size`), and the workload runs in the shipped tree unless --cwd says otherwise (its environment is the workload's to
    sync, as suites.py does). Fills `ns` and `harness_opts` in place; a reason when it can't."""
    if ns.queue or ns.pool:
        return "it takes no --queue or --pool: it is the queue itself"
    from .auto import AUTO_NODE, AUTO_PROJECT

    if not ns.source:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if top.returncode != 0:
            return "--source GITDIR: no git checkout here to ship"
        ns.source = top.stdout.strip()
    ns.queue, ns.on, ns.preemptible = True, AUTO_NODE, True      # a guest on node 2 is preemptible; a test reruns from the start
    ns.auto = True
    ns.project = ns.project or AUTO_PROJECT
    if remote.flag_value(harness_opts, "--cwd") is None:
        harness_opts += ["--cwd", "source"]                      # a suite or build runs in the tree it tests (INF-65), not its run dir
    if ns.queue_kind == "adhoc":
        ns.queue_kind = AUTO_KIND
    return None


def _auto_spill(ns: argparse.Namespace, harness_opts: list[str]) -> list[str] | None:
    """`--on auto` once sized: queued onto AUTO_NODE as `_auto_placement` set it, or, where research.auto finds it starts sooner,
    a CPU-phase run of AUTO_KIND sent to a pool instead (`--on MACHINE --pool POOL`, under `--as`, else `auto`), with the
    kind's time as its --timeout. The harness options it runs with, or None when it is refused (`auto.Unplaceable`)."""
    from . import auto

    cpus = ns.cpus or AUTO_CPUS
    mem = ns.mem_gb if ns.mem_gb is not None else AUTO_MEM_GB
    gpu = ns.gpus or ns.gpu_model or ns.gpu_ids or ns.gpu_host or ns.quiet or ns.queue_phase == "gpu"
    target, why = None, "a GPU or quiet job, which only the queue books" if gpu else None
    if why is None:
        try:
            target, why = auto.choose(auto.read(Path(ns.source), cpus), cpus, mem)
        except auto.Unplaceable as e:
            print(f"research run --on auto: refused: {e}", file=sys.stderr)
            return None
    if target is None:
        print(f"research run --on auto: queued onto {auto.AUTO_NODE} as {ns.queue_kind}, from {ns.source} ({why})", file=sys.stderr)
        return harness_opts
    machine, pool = target
    if remote.flag_value(harness_opts, "--timeout") is None:
        harness_opts = [*harness_opts, "--timeout", f"{(ns.max_min or AUTO_MAX_MIN) * 60}"]
    ns.queue, ns.on, ns.pool, ns.lane = False, machine, pool, ns.lane or "auto"
    ns.cpus = ns.max_min = ns.question = ns.disk_gb = None     # the slot's cores are the run's
    ns.mem_gb = mem     # --pool hands it to the slot, which reserves the larger of its mem= and this (INF-228: a 40-GB slot held a suite to 2 workers)
    ns.queue_kind, ns.queue_class, ns.queue_phase, ns.preemptible = "adhoc", "work", None, False
    print(f"research run --on auto: spilled to {machine}'s {pool} pool, from {ns.source} ({why})", file=sys.stderr)
    return harness_opts


def _auto_size(ns: argparse.Namespace, harness_opts: list[str], command: list[str]) -> None:
    """`--on auto` as AUTO_KIND, of a suites.py run (named suites, `--changed REV`'s or all): the memory and time their
    `[tool.verity.tests]` worker_gb and timeout_s call for (research.suite_size), where it gives no --mem-gb, or no --timeout or
    --max-min."""
    timed = ns.max_min is not None or remote.flag_value(harness_opts, "--timeout") is not None
    if ns.queue_kind != AUTO_KIND or (ns.mem_gb is not None and timed):
        return
    from . import suite_size

    if not (shape := suite_size.size(Path(ns.source), command, ns.cpus)):
        return
    said = []
    if ns.mem_gb is None and "mem_gb" in shape:
        ns.mem_gb = shape["mem_gb"]
        said.append(f"{ns.mem_gb} GB")
    if not timed and "max_s" in shape:
        ns.max_min = math.ceil(shape["max_s"] / 60)
        said.append(f"{ns.max_min} min")
    if said:
        of = shape["suites"]
        of = ", ".join(of) if len(of) <= 6 else f"its {len(of)} suites ({', '.join(of[:3])}, ...)"
        print(f"research run --on auto: {' and '.join(said)}, from the [tool.verity.tests] of {of}", file=sys.stderr)


def _run_queued(ns: argparse.Namespace, harness_opts: list[str], command: list[str]) -> int:
    """`research run --queue`: the queue places the job (`cluster submit`), then it launches as `--on` the placed machine does, with
    the runner inside the job's CPU allocation and a gpu phase's workload alone inside its GPU lease. Its records, status and logs
    are an `--on` run's."""
    if not ns.source or not ns.project:
        print("research run --queue: --source GITDIR (the tree with infra/cluster, whose kinds/ it resolves) and --project are "
              "required", file=sys.stderr)
        return 2
    timeout = remote.timeout_seconds(remote.flag_value(harness_opts, "--timeout"))
    from .pods.lease import RUN_MARGIN_S

    root = R.runs_root(ns.runs_dir)
    run_id = ns.id or R.new_run_id()
    given_max = ns.max_min * 60 if ns.max_min is not None else timeout
    spec = {"project": ns.project, "submitter": os.environ.get("RESEARCH_LANE") or os.environ.get("USER") or "-",
            "class": ns.queue_class, "kind": ns.queue_kind, "question": ns.question or None, "gpus": ns.gpus,
            "gpu-model": ns.gpu_model, "cpus": ns.cpus, "mem-gb": math.ceil(ns.mem_gb) if ns.mem_gb is not None else None,
            "max-s": int(given_max) if given_max else None, "custody-s": RUN_MARGIN_S,
            "disk-gb": ns.disk_gb,
            "preempt": "requeue" if ns.preemptible else "never", "quiet": ns.quiet, "node": ns.on, "gpu-ids": ns.gpu_ids,
            "gpu-host": ns.gpu_host}
    try:
        placed = remote.queue_placement(Path(ns.source).expanduser(), run_id, spec)
    except remote.Refused as e:
        print(f"research run --queue: refused: {e}", file=sys.stderr)
        return 3
    gpus = int(placed["job"].get("gpus", 0))
    phase = ns.queue_phase or ("gpu" if gpus else "cpu")
    if (phase == "gpu") != bool(gpus):
        print(f"research run --queue: a {phase} phase {'needs GPUs' if phase == 'gpu' else 'takes no GPUs'} (the job has {gpus})",
              file=sys.stderr)
        return 2
    if timeout is None:
        harness_opts = [*harness_opts, "--timeout", str(int(placed.get("max_s") or given_max) + (GPU_WAIT_MIN * 60 if gpus else 0))]
    placed.update(phase=phase)
    placed.setdefault("disk_gb", ns.disk_gb if ns.disk_gb is not None else 20)
    placed.setdefault("kind", ns.queue_kind)
    print(f"research: the queue places run {run_id} ({placed['kind']}, {phase} phase) on {placed['machine']} ({placed['why']})",
          file=sys.stderr)
    for w in placed.get("warnings") or ():
        print(f"research run --queue: warning: {w}", file=sys.stderr)
    ns.on, ns.mem_gb = placed["machine"], None
    if ns.custody_r2 is not False and (why := _custody_ttl_problem(ns.custody_ttl, harness_opts)):
        print(f"research run: {why}", file=sys.stderr)
        return 2
    harness_opts = ["--project", ns.project, *harness_opts]
    return _run_remote(ns, run_id, root / run_id, harness_opts, command, queue=placed)


def _run_remote(ns: argparse.Namespace, run_id: str, run_dir: Path, harness_opts: list[str], command: list[str],
                queue: dict | None = None) -> int:
    if not ns.project:
        print("research run --on: --project is required (the machine is bound to a project)", file=sys.stderr)
        return 2
    if run_dir.exists():
        print(f"research run: {run_dir} already exists; a rerun gets a fresh id", file=sys.stderr)
        return 2
    env = {}
    for kv in ns.env:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            print(f"research run: --env wants K=V, got {kv!r}", file=sys.stderr)
            return 2
        env[k] = v
    if queue:
        cpus = str(int((queue.get("job") or {}).get("cpus") or 1))
        env = {**dict.fromkeys(THREAD_POOL_VARS, cpus), **env}
    asked_cwd = remote.flag_value(harness_opts, "--cwd")
    if asked_cwd is not None and not os.path.isabs(asked_cwd) and not ns.source:
        print(f"research run --on {ns.on}: --cwd {asked_cwd} is relative to the shipped source tree, and there is none; pass --source GITDIR "
              f"or an absolute --cwd (the default is the run dir)", file=sys.stderr)
        return 2
    if asked_cwd is None and ns.source and command and "/" in command[0] and not os.path.isabs(command[0]):
        print(f"research run --on {ns.on}: {command[0]} is relative, and without --cwd the workload starts in the run dir, not the shipped "
              f"tree; pass --cwd source (or --cwd DIR)", file=sys.stderr)
        return 2
    harness_args = ["--stage", ns.stage, *harness_opts]
    for e in ns.expect:
        harness_args += ["--expect", e]
    if ns.require_result:
        harness_args.append("--require-result")
    # the store flags travel as `research run` flags: the machine-side runner computes derivation/reuse_key in the shipped source tree
    if ns.tool:
        # the machine runs the shipped tree's research package when --source carries one, else THIS launcher's: that one must know the tool
        from .store import tools_registry
        if remote.source_tool_package(Path(ns.source).expanduser() if ns.source else None) is None and ":" not in ns.tool \
                and ns.tool not in tools_registry.REGISTRY:
            print(f"research run --on {ns.on}: launcher research package does not know tool {ns.tool!r}; pass --source with tools/research "
                  f"(registered here: {', '.join(sorted(tools_registry.REGISTRY))})", file=sys.stderr)
            return 3
        harness_args += ["--tool", ns.tool]
    for s in ns.scratch:
        harness_args += ["--scratch", s]
    if ns.campaign:
        harness_args += ["--campaign", ns.campaign]
    custody = ns.custody_r2 is not False
    if not custody:
        print(f"research run --on {ns.on}: --no-custody-r2: this run's records stay on the machine until `research fetch --all {run_id}`",
              file=sys.stderr)
    try:
        rec = remote.launch(run_id=run_id, local_dir=run_dir, machine_name=ns.on, project=ns.project, command=command, harness_args=harness_args,
                            source=Path(ns.source).expanduser() if ns.source else None, sends=[Path(s).expanduser() for s in ns.send],
                            cwd=None, env=env, exclusive=ns.exclusive, mem_gb=ns.mem_gb, mem_wait_s=ns.mem_wait,
                            custody_ttl=(ns.custody_ttl or "6h") if custody else None, preflight_tool=ns.preflight, queue=queue,
                            job_user=ns.job_user, gpus=None if queue else ns.gpus, store_read=ns.store_read)
    except remote.Refused as e:
        print(f"research run --on {ns.on}: refused: {e}", file=sys.stderr)
        return 3
    launch = rec.get("launch") or {}
    if rec["phase"] == "refused":
        print(f"research: run {run_id} refused on {ns.on} ({launch.get('fault') or 'Refused'}): {launch.get('refused')}  (recorded in {run_dir})", file=sys.stderr)
        return 3
    print(f"research: run {run_id} launched on {ns.on} ({rec['pod_id']}) pid={launch.get('runner_pid')} -> {rec['remote_run_dir']}", file=sys.stderr)
    if launch.get("unconfirmed"):
        print(f"research: {ns.on} could not confirm its runner is up ({launch['unconfirmed']}); `research fetch {run_id}` says how it went",
              file=sys.stderr)
    if launch.get("first_record") == "disk-hold":
        print(f"research: {ns.on} holds it for disk before its runner starts (research disk-hold, in its launcher.log)", file=sys.stderr)
    if launch.get("first_record") == "cpu-hold":
        print(f"research: {ns.on} holds it until whole cores of its own are free (research cpu-hold, in its launcher.log)", file=sys.stderr)
    print(f"research: workload cwd: {rec.get('workload_cwd') or remote.WORKLOAD_CWD_DEFAULT}", file=sys.stderr)
    if getattr(ns, "wait", False):
        launched_s = time.monotonic() - getattr(ns, "started", time.monotonic())
        return wait_remote(run_id, run_dir, every_s=ns.wait_every, max_s=max(1.0, waitcap.max_s(ns.wait_max) - launched_s))
    print(f"research: observe with `research fetch {run_id}`; the workload does not depend on this session staying alive", file=sys.stderr)
    return 0


#: the states `summarize` gives a remote run that will not change again
ENDED = (*R.TERMINAL, "refused", "launcher-exited", "runner-gone")


def wait_remote(run_id: str, run_dir: Path, *, every_s: float, max_s: float = waitcap.WAIT_CAP_S, fetch=None, sleep=None,
                clock=None) -> int:
    """`research run --on ... --wait`: fetch the run's records every `every_s` seconds until it has ended, and return its rc (1 when
    it ended without one, 3 when the machine refused it). An unreachable machine is waited out, as `research fetch` would be, but
    no wait outlasts `max_s` of `clock` time: then the run goes on, the check-back command is printed, and it returns
    STILL_GOING (research.waitcap)."""
    import time

    sleep, clock = sleep or time.sleep, clock or time.monotonic
    start = clock()
    fetch = fetch or (lambda d: remote.fetch(d, timeout=max_s - (clock() - start)))
    while True:
        rec = fetch(run_dir)
        s = R.summarize(run_dir)
        if s["state"] in ENDED:
            if s["state"] == "refused":
                rc = 3
            elif s["state"] == "done":
                rc = s["rc"] if isinstance(s["rc"], int) else 0
            else:
                rc = s["rc"] if isinstance(s["rc"], int) and s["rc"] != 0 else 1
            why = f": {s['why']}" if s.get("why") else ""
            print(f"research: run {run_id} {s['state']} (rc {s['rc']}){why}", file=sys.stderr)
            return rc
        if (rec or {}).get("last_fetch_error"):
            print(f"research: run {run_id}: machine unreachable (rc {rec['last_fetch_error'].get('rc')}); still waiting", file=sys.stderr)
        if (left := max_s - (clock() - start)) <= 0:
            print(waitcap.still_going(f"run {run_id} ({s['state']})", f"research status {run_id}", clock() - start), file=sys.stderr)
            return waitcap.STILL_GOING
        sleep(min(every_s, left))


def _run_dir(root: Path, ref: str, on: str | None, *, unique: bool = False) -> Path:
    """`ref` under the local runs root, else, for a full run id, the run another VM launched, adopted from the machine that holds
    it (`remote.adopt`; `unique`: refused when two machines hold it).  FileNotFoundError names each machine's answer, and the store's
    attempt when it has one; LookupError when `on` names another machine than the local run dir's."""
    try:
        d = R.resolve(root, ref)
    except FileNotFoundError:
        from .store import ids

        if not ids.is_run_id(ref):
            raise
    else:
        if on and (held := (remote._read_json(d / "remote.json") or {}).get("machine")) != on:
            raise LookupError(f"{d.name} is {f'on {held}' if held else 'a local run'} (its local run dir), not on {on}")
        return d
    try:
        return remote.adopt(root, ref, on=on, unique=unique)
    except remote.Refused as e:
        raise FileNotFoundError(f"no run {ref!r} under {root}, and {e}{_attempt_hint(ref)}") from None


def _attempt_hint(run_id: str) -> str:
    try:
        from .store.local import LocalStore

        found = LocalStore().get_attempt(run_id) is not None
    except Exception:  # noqa: BLE001  no store, no remote: the hint is optional
        return ""
    return f"; the store has its attempt: `research data show {run_id}`" if found else ""


def cmd_fetch(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research fetch")
    ap.add_argument("ref")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--on", default=None, metavar="NAME", help="a run another VM launched: look for it on this machine only")
    ap.add_argument("--all", action="store_true", help="every file, verified by sha256; writes preserved.json last")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    if ns.all and (why := R.disk_floor_refusal(root)):          # the records alone are small: observing a run stays possible
        print(f"research fetch --all: refused: {why}", file=sys.stderr)
        return 3
    try:
        run_dir = _run_dir(root, ns.ref, ns.on)
        rec = remote.fetch(run_dir, everything=ns.all)
    except (FileNotFoundError, LookupError, remote.Refused) as e:
        print(f"research fetch: {e}", file=sys.stderr)
        return 1
    if rec.get("last_fetch_error"):
        lo = rec.get("last_observed") or {}
        print(f"research fetch: machine unreachable ({rec['last_fetch_error']['stderr'].strip() or 'rc ' + str(rec['last_fetch_error']['rc'])}); "
              f"last observed {lo.get('state')} at {lo.get('at_utc')}; current status unknown", file=sys.stderr)
        return 4
    s = R.summarize(run_dir)
    if ex := rec.get("launcher_exited"):
        print(f"research fetch: {s['run_id']}: the machine's runner (pid {ex['runner_pid']}) exited without writing status.json; "
              f"launcher.log ends: {ex['last_line'] or '(empty)'}", file=sys.stderr)
    if gone := rec.get("runner_gone"):
        print(f"research fetch: {s['run_id']}: it ended without recording it: {gone['why']}", file=sys.stderr)
    print(f"research: {s['run_id']} observed {rec['observed_utc']}: {s['state']} rc={s['rc']} class={s['class'] or '-'} result={s['result']}"
          + (f"  preserved={'yes' if rec.get('preserved') else 'NO: ' + json.dumps(rec.get('preserve_error'))}" if ns.all else ""))
    marker = rec.get("fetched_marker") if ns.all else None
    if marker and not marker.get("written"):
        print(f"research: {s['run_id']}: .fetched NOT written on the machine ({marker.get('reason')}); a pod guard keeps refusing to terminate",
              file=sys.stderr)
    return 0


def cmd_cancel(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research cancel", description="stop a remote run the one way that keeps its custody: SIGTERM to its "
                                                                     "workload's process group and its whole cgroup scope, recorded as a "
                                                                     "manual cancel; the runner SIGKILLs what is left 20 s later "
                                                                     "(remote.cancel). A run that hasn't started, held on "
                                                                     "its machine (disk-hold, cpu-hold), is withdrawn: its hold "
                                                                     "ends and refused.json records it Withdrawn")
    ap.add_argument("ref", help="a run id or prefix under the runs root; a full run id another VM launched is found on its machine")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE") or os.environ.get("USER") or "-")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--on", default=None, metavar="NAME", help="a run another VM launched: look for it on this machine only")
    ns = ap.parse_args(argv)
    try:
        run_dir = _run_dir(R.runs_root(ns.runs_dir), ns.ref, ns.on, unique=True)
        out = remote.cancel(run_dir, reason=ns.reason, by=ns.by)
    except (FileNotFoundError, LookupError, remote.Refused) as e:
        print(f"research cancel: {e}", file=sys.stderr)
        return 1
    if out.get("withdrawn"):
        holds = ", ".join(f"research {h['hold']} pid {h['pid']}" for h in out["holds"])
        print(f"research cancel: {run_dir.name}: withdrawn before it started: SIGTERM to its hold ({holds}), recorded as a manual "
              f"cancel; refused.json records it {(out.get('refused') or {}).get('fault')}")
        if out.get("released"):
            print(f"research cancel: {run_dir.name}: removed what its launch staged: {', '.join(out['released'])}")
        return 0
    if not out["delivered"]:
        print(f"research cancel: {run_dir.name}: SIGTERM to process group {out['pgid']} not delivered ({out['why']})", file=sys.stderr)
        return 1
    print(f"research cancel: {run_dir.name}: SIGTERM to its workload's process group {out['pgid']} ({out['processes']} process(es)), "
          f"recorded as a manual cancel; the runner records the exit and publishes custody (`research fetch {run_dir.name}`)")
    scope = out.get("scope") or {}
    if scope.get("scopes"):
        print(f"research cancel: {run_dir.name}: and to the {(scope.get('SIGTERM') or {}).get('n', 0)} other process(es) of its scope(s) "
              f"{', '.join(p.rsplit('/', 1)[-1] for p in scope['scopes'])}")
    if scope.get("fallback"):
        print(f"research cancel: {run_dir.name}: {scope['fallback']}")
    if scope.get("runner_kills_after_s") is not None:
        print(f"research cancel: {run_dir.name}: the runner SIGKILLs what is left of the run {scope['runner_kills_after_s']:g} s after the "
              f"SIGTERM and records it in status.json (`scope_end`)")
    elif scope.get("cancel_killed_after_s") is not None:
        left = scope.get("left") or []
        print(f"research cancel: {run_dir.name}: {scope.get('why')}, so the cancel SIGKILLed what was left of the run "
              f"{scope['cancel_killed_after_s']:g} s after the SIGTERM ({(scope.get('SIGKILL') or {}).get('n', 0)} process(es)); "
              + (f"still there: {', '.join(left)}" if left else "none is left") + f"; status.json `scope_end` {scope.get('status_json')}")
    if ran := _ran_text(R.summarize(run_dir), datetime.now(timezone.utc)):
        print(f"research cancel: {run_dir.name} {ran}")
    return 0


def _utc(text: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc) if text else None
    except ValueError:
        return None


def _ran_text(s: dict, now: datetime) -> str:
    """How long a run ran, and how much of that its check waited for a check slot (runs.SLOT_FILE): its state reads `running` from
    launch, so a check that waited 95 min for a slot and checked 12 was taken for 107 min in (af2d, 10 Oct)."""
    if not (start := _utc(s.get("started_utc"))):
        return ""
    sl, mins = s.get("slot") or {}, lambda a, b: f"{max(0.0, (b - a).total_seconds()) / 60:.0f} min"
    text = f"ran {mins(start, now)} since {start:%Y-%m-%dT%H:%MZ}"
    since, took = _utc(sl.get("since")), _utc(sl.get("took"))
    if sl.get("state") == "held" and since and took:
        return f"{text}: waited {mins(since, took)} for a check slot, then checked {mins(took, now)} in slot {sl.get('slot')}"
    if sl.get("state") == "waiting" and since:
        return f"{text}: waiting for a check slot since {since:%H:%MZ} ({mins(since, now)}), so its check never started"
    if sl.get("state") == "left":
        return f"{text}: left the check line without a slot ({sl.get('why')}), so its check never started"
    return text


def cmd_compare(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="research compare", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="ORIGINAL... REPLAY: the replay's verdict against the originals, read from the evidence store by run id (local, "
                    "else its remote), so the run dirs need not exist here. Exit 0 for same, within-band and expected-change, 1 for "
                    "differs. --dirs (or a run the store lacks): the descriptive diff of two run dirs under --runs-dir.",
        epilog=C.__doc__)
    ap.add_argument("runs", nargs="+", metavar="RUN", help="run ids: one or more originals, then the replay")
    ap.add_argument("--expect", default=None, help="a replay_expect value (`refused: FORM`, `blocked: GATE`); default: the latest "
                                                   "replay_expect label on the first original, if any")
    ap.add_argument("--no-expect", action="store_true", help="ignore the originals' replay_expect labels")
    ap.add_argument("--store", default=None, help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--dirs", action="store_true", help="the descriptive diff of two run directories (BASELINE CANDIDATE)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    if len(ns.runs) < 2:
        ap.error("needs at least one original and the replay")
    if not ns.dirs:
        from .store import ids as store_ids
        from .store.local import LocalStore, StoreError

        st = LocalStore(ns.store)
        try:
            bad = [r for r in ns.runs if not store_ids.is_run_id(r)]
            if bad:
                raise LookupError(f"{bad[0]}: not a full run id")
            atts = [C.load(st, r) for r in ns.runs]
            expect = None if ns.no_expect else ns.expect
            if expect is None and not ns.no_expect:
                labels = st.labels(ns.runs[0], key=C.EXPECT_KEY)
                expect = str(labels[-1].value) if labels else None
            v = C.verdict(atts[:-1], atts[-1], expect)
        except LookupError as e:
            if len(ns.runs) != 2:
                print(f"research compare: {e}", file=sys.stderr)
                return 1
            print(f"research compare: {e}; comparing the run dirs instead", file=sys.stderr)
        except StoreError as e:
            print(f"research compare: {e}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"research compare: {e}", file=sys.stderr)
            return 2
        else:
            print(json.dumps(v.to_doc(), indent=2, sort_keys=True) if ns.json else C.render_verdict(v))
            return 1 if v.verdict == C.DIFFERS else 0
        finally:
            st.close()
    if len(ns.runs) != 2:
        ap.error("--dirs compares exactly two run dirs")
    root = R.runs_root(ns.runs_dir)
    try:
        b, c = R.resolve(root, ns.runs[0]), R.resolve(root, ns.runs[1])
    except (FileNotFoundError, LookupError) as e:
        print(f"research compare: {e}", file=sys.stderr)
        return 1
    rep = C.compare(b, c)
    print(json.dumps(rep, indent=2, sort_keys=True, default=str) if ns.json else C.render(rep))
    return 0


def cmd_env(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research env", description="the PYTHONPATH a source tree's workspace declares (research.pythonpath)")
    ap.add_argument("--source", required=True, metavar="TREE", help="a checkout or shipped tree with a root pyproject.toml [tool.uv.workspace]")
    ap.add_argument("--rev", default=None, help="read this git revision of TREE instead of its files")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    from .pythonpath import import_roots_dir, import_roots_git, pythonpath

    tree = Path(ns.source).expanduser().resolve()
    try:
        derived = import_roots_git(tree, ns.rev) if ns.rev else import_roots_dir(tree)
    except LookupError as e:
        print(f"research env: {e}", file=sys.stderr)
        return 1
    if ns.json:
        print(json.dumps({**derived, "tree": str(tree), "pythonpath": pythonpath(str(tree), derived)}, indent=2))
    else:
        print(f"export PYTHONPATH={pythonpath(str(tree), derived)}")
    for m in derived["missing"]:
        print(f"research env: declared but absent: {m}", file=sys.stderr)
    if not derived["roots"]:
        print(f"research env: {tree}: no import roots ({derived['source']})", file=sys.stderr)
        return 1
    return 0


def cmd_launch_request(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research launch-request")
    ap.add_argument("--expect-tool", default=None, metavar="IDENTITY", help="the launcher's remote.tool_identity of the package it shipped")
    ap.add_argument("request")
    ns = ap.parse_args(argv)
    try:
        remote.check_tool(ns.expect_tool)                 # before the claim: a mismatched package must not interpret the request
        out = remote.launch_request(Path(ns.request))
    except remote.Refused as e:
        print(json.dumps({"claimed": False, "refused": str(e), "fault": type(e).__name__}))
        return 3
    print(json.dumps(out))
    return 0


def _read_holds(rows: list[dict]) -> None:
    """Each live run's `scope_memory` and `holds` (research.holds' lines) where it executes, one ssh per machine for remote runs,
    or its `scope_memory_error`."""
    from . import holds, scopemem

    remotes = [Path(s["dir"]) for s in rows if s.get("remote_rec")]
    got = remote.node_readings(remotes) if remotes else {}
    for s in rows:
        if s.get("remote_rec"):
            r = got.get(str(Path(s["dir"]))) or {"error": "not read"}
        else:
            doc = scopemem.node([str(s["dir"])])
            r = {"run": doc.pop("runs")[str(s["dir"])], "node": doc, "machine": None}
        if r.get("error"):
            s["scope_memory_error"] = r["error"]
            continue
        s["scope_memory"] = r["run"]["memory"]
        s["holds"] = holds.lines(s.get("run_id") or s["id"], r["machine"], r)


def _fmt(s: dict) -> str:
    cmd = " ".join(s["command"]).replace("\n", " ") if s.get("command") else "?"
    if len(cmd) > 50:
        cmd = cmd[:47] + "..."
    rc = "" if s["rc"] is None else str(s["rc"])
    res = "-" if not s["result_present"] else ("ok" if s["result_valid"] else "BAD")
    val = s["validation"] or "-"
    where = (s.get("remote") or {}).get("machine") or "local"
    return f"{s['id']:<24} {str(s['state']):<10} {rc:>3}  {(s['class'] or ''):<22} {res:<4} {val:<8} {str(s['project']):<10} {where:<11} {str(s['stage']):<16} {cmd}"


def cmd_status(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research status")
    ap.add_argument("ref", nargs="?", help="just this run, its records fetched first while it is remote and not finished (a full run "
                                           "id another VM launched is found on its machine)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--on", default=None, metavar="NAME", help="with RUN another VM launched: look for it on this machine only")
    ap.add_argument("--all", action="store_true", help="every run, not just the last 20")
    ap.add_argument("--refresh", action="store_true", help="fetch the records of every non-terminal remote run first, and read each "
                                                          "active one's memory on its machine (research.scopemem)")
    ap.add_argument("--balance", action="store_true", help="show the provider's account ledger (balance, spend limit, current $/h) with its observation time")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--also", action="append", default=[], metavar="FILE",
                    help="with RUN: fetch this file of its run dir with its records (repeatable): a workload's own progress file")
    ap.add_argument("--basis", default=None, metavar="RUN",
                    help="with RUN, a check: time its steps against this finished check (default: the newest other finished "
                         "check on its machine)")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    ledger = None
    if ns.balance:
        try:
            from .pods import runpod
            from .telemetry.events import utc_now_ms

            data = runpod._graphql("{ myself { clientBalance spendLimit currentSpendPerHr } }")["data"]["myself"]
            ledger = {"provider": "runpod", "observed_utc": utc_now_ms(), **data}
        except Exception as e:  # noqa: BLE001
            ledger = {"provider": "runpod", "error": f"{type(e).__name__}: {e}"}
    one = None
    if ns.ref:
        try:
            one = _run_dir(root, ns.ref, ns.on)
        except (FileNotFoundError, LookupError) as e:
            print(f"research status: {e}", file=sys.stderr)
            return 1
    if ns.refresh or one:
        for s in [R.summarize(one)] if one else R.list_runs(root):
            if s.get("remote_rec") and (s["state"] not in R.TERMINAL or (one and (ns.also or not (Path(s["dir"]) / checkview.SPANS).exists()))) \
                    and (s["remote_rec"].get("remote_run_dir")):
                try:
                    remote.fetch(Path(s["dir"]), also=(*ns.also, *checkview.CHECK_FILES) if one else ())
                except remote.Refused as e:
                    print(f"research status: {s['id']}: {e}", file=sys.stderr)
    rows = [R.summarize(one)] if one else R.list_runs(root)
    if not ns.all:
        rows = rows[-20:]
    unread = 0
    live = []
    for s in rows:
        rr = s.get("remote_rec")
        if s["state"] not in R.ACTIVE or (rr and not rr.get("remote_run_dir")):
            continue
        if rr and not (ns.refresh or one):
            unread += 1
            continue
        live.append(s)
    _read_holds(live)
    check = _check_view(root, one, ns.basis) if one and checkview.is_check(one) else None
    if ns.json:
        print(json.dumps({"runs_root": str(root), "runs": rows, "ledger": ledger, **({"check": check} if check else {})},
                         indent=2, sort_keys=True))
        return 0
    if ledger:
        if ledger.get("error"):
            print(f"ledger    unavailable: {ledger['error']}")
        else:
            print(f"ledger    runpod balance ${ledger['clientBalance']:.2f}  spend limit ${ledger['spendLimit']}  now ${ledger['currentSpendPerHr']}/h  (observed {both(ledger['observed_utc'])}; the provider's numbers, not a research counter)")
        print()
    if not rows:
        print(f"no runs under {root}")
        return 0
    print(f"{'run':<24} {'state':<10} {'rc':>3}  {'class':<22} {'res':<4} {'valid':<8} {'project':<10} {'where':<11} {'stage':<16} command")
    for s in rows:
        print(_fmt(s))
        for label, text in s.get("holds") or []:
            print(f"{'':<26}{label:<4} {text}")
        if s.get("scope_memory_error"):
            print(f"{'':<26}mem  unread: {s['scope_memory_error']}")
        for line in check or []:
            print(f"{'':<26}{line}")
    active = sum(1 for s in rows if s["state"] in R.ACTIVE)
    unfetched = sum(1 for s in rows if s.get("remote_rec") and not s["has_job"] and s["state"] != "refused")
    print(f"\n{len(rows)} shown ({active} active" + (f", {unfetched} remote not yet fetched: `research status --refresh`" if unfetched else "")
          + (f", memory of {unread} active remote run(s): `--refresh` or `research status RUN`" if unread else "") + f") under {root}"
          f"   res: result.json ok/BAD/-absent   valid: the workload's own checker")
    return 0


def _check_view(root: Path, run_dir: Path, basis_ref: str | None) -> list[str]:
    """`checkview.view` of a check run: against ``basis_ref`` (a finished check, fetched when it is remote), else the newest
    other finished check on the run's machine, else none; its first line names the basis."""
    basis, why = None, ""
    if basis_ref:
        try:
            basis = _run_dir(root, basis_ref, None)
        except (FileNotFoundError, LookupError) as e:
            why = f"--basis {basis_ref}: {e}"
        else:
            if (remote._read_json(basis / "remote.json") or {}).get("remote_run_dir") and not checkview.finished(basis):
                try:
                    remote.fetch(basis, also=checkview.CHECK_FILES)
                except remote.Refused as e:
                    why = f"--basis {basis_ref}: {e}"
            if not checkview.finished(basis):
                basis, why = None, why or f"--basis {basis_ref} is no finished check"
    elif not checkview.finished(run_dir) and (rec := remote._read_json(run_dir / "remote.json") or {}).get("remote_run_dir"):
        ssh = remote.ssh_argv(rec["host"], rec["port"], user=rec.get("user") or "root")
        basis = checkview.remote_basis(rec, ssh, root / ".check-basis", remote.ssh_read)
    head = [] if checkview.finished(run_dir) and not basis_ref else [
        f"basis   {basis.name}" if basis else f"basis   none{f' ({why})' if why else ''}"]
    return head + checkview.view(run_dir, datetime.now(timezone.utc), basis)


def _slot_text(sl: dict) -> str:
    """A run's check-slot record (runs.SLOT_FILE) in a line: waiting since when and why, the slot it took and when, or why it left."""
    if sl.get("state") == "held":
        return f"check slot {sl.get('slot')} (cores {sl.get('cores')}) taken {sl.get('took')}, after waiting in line since {sl.get('since')}"
    if sl.get("state") == "left":
        return f"left the check line at {sl.get('left')} without a slot, waiting since {sl.get('since')}: {sl.get('why')}"
    return f"waiting in the check line since {sl.get('since')}, not started" + (f": {sl['why']}" if sl.get("why") else "")


def _tool_line(tool: dict | None) -> str:
    if tool is None:
        return "unknown (v0.1 record: the harness alone wrote it)"
    sha = tool.get("research_git_sha")
    state = "" if tool.get("research_dirty") is None else (" (dirty)" if tool.get("research_dirty") else " (clean)")
    ident = f"git {sha[:12]}{state}" if sha else (f"snapshot {tool['research_snapshot']}" if tool.get("research_snapshot") else "no git sha, no snapshot")
    return f"research {tool.get('research_version')} @ {ident}; classifier {tool.get('classifier_version')}"


def cmd_inspect(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research inspect")
    ap.add_argument("ref", help="run id, unique prefix, or directory; a full run id another VM launched is fetched from its machine")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--on", default=None, metavar="NAME", help="a run another VM launched: look for it on this machine only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--py-spy", action="store_true", help="instead: print a py-spy dump of the live run's Python processes, read on its "
                                                          "machine (research.telemetry.pyspy)")
    ns = ap.parse_args(argv)
    root = R.runs_root(ns.runs_dir)
    try:
        run_dir = _run_dir(root, ns.ref, ns.on)
    except (FileNotFoundError, LookupError) as exc:
        print(f"research inspect: {exc}", file=sys.stderr)
        return 1
    if ns.py_spy:
        return _inspect_py_spy(run_dir)
    s = R.summarize(run_dir)
    if ns.json:
        print(json.dumps(s, indent=2, sort_keys=True))
        return 0
    print(f"run       {s['run_id']}   ({s['schema'] or 'no job.json'})")
    print(f"dir       {s['dir']}")
    if not s["readable"]:
        print(f"WARNING   job.json schema {s['schema']!r} is not one this research can read; fields below are best effort")
    print(f"exited    {s['state']}  rc={s['rc']} signal={s['signal']} wall_s={s['wall_s']}")
    if s["class"]:
        print(f"class     {s['class']} ({s['certainty']})" + (f": {s['why']}" if s["why"] else ""))
    print(f"result    {s['result']}")
    print(f"validated {s['validation'] or 'no validation reported'}  (the workload's checker; research does not verify proofs)")
    pres = s["preserved"]
    print(f"preserved {'yes: ' + str(pres.get('destination')) if pres else 'no preservation manifest'}")
    print(f"project   {s['project']}  lane={s['lane']}  stage={s['stage']}")
    print(f"source    tree_sha={s['tree_sha']}  workload_digest={s['workload_digest']}")
    if s["fingerprint_expected"] is not None or s["fingerprint_observed"] is not None:
        print(f"expected  {json.dumps(s['fingerprint_expected'], sort_keys=True)}")
        print(f"observed  {json.dumps(s['fingerprint_observed'], sort_keys=True)}")
    if s["measurements"]:
        print("measured  " + "; ".join(f"{m['name']}={m['value']} {m['unit']}" for m in s["measurements"][:12]))
    print(f"tool      {_tool_line(s['tool'])}")
    cost = s["cost"]
    print(f"cost      {('$' + str(cost['usd']) + ' est. (' + cost['basis'] + ')') if cost.get('usd') is not None else 'unknown: ' + str(cost.get('reason'))}")
    print(f"time      started={s['started_utc']}  ended={s['ended_utc']}")
    print(f"command   {' '.join(s['command']) if s['command'] else '?'}")
    if s["remote"]:
        r = s["remote"]
        print(f"remote    {r.get('machine')} ({r.get('pod_id')}) {r.get('root')}  tool={str(r.get('tool')).rsplit('/', 1)[-1]}  source={r.get('source_sha')}"
              + (f"  archive={str(r.get('source_archive_sha256'))[:16]} ({r.get('source_verified')})" if r.get("source_archive_sha256") else ""))
    if s.get("remote_rec"):
        rr = s["remote_rec"]
        lo = rr.get("last_observed") or {}
        print(f"observed  phase={rr.get('phase')} at={rr.get('observed_utc')} state={lo.get('state')}"
              + (f"  ERROR={rr.get('error')}" if rr.get("error") else "")
              + ("  (records shown are a cached copy; `research fetch` refreshes)" if rr.get("observed_utc") else "  (nothing fetched yet)"))
        if q := rr.get("queue"):
            print(f"queue     placed on {q.get('machine')} ({q.get('why')}); {q.get('phase')} phase")
    if sl := s.get("slot"):
        print(f"slot      {_slot_text(sl)}")
    if s["state"] in R.ACTIVE and (not s.get("remote_rec") or s["remote_rec"].get("remote_run_dir")):
        row = {**s, "id": Path(s["dir"]).name}
        _read_holds([row])
        for label, text in row.get("holds") or []:
            print(f"{label:<9} {text}")
        if row.get("scope_memory_error"):
            print(f"mem       unread: {row['scope_memory_error']}")
    files = sorted(p.name for p in Path(s["dir"]).iterdir() if p.is_file())
    print(f"files     {', '.join(files)}")
    missing = [n for n, ok in (("job.json", s["has_job"]), ("status.json", s["has_status"])) if not ok]
    if missing:
        print(f"missing   {', '.join(missing)}  (the run died before the harness wrote them)")
    return 0


def _inspect_py_spy(run_dir: Path) -> int:
    from .telemetry import pyspy

    if (run_dir / "remote.json").exists():
        try:
            text = remote.py_spy(run_dir)
        except remote.Refused as e:
            print(f"research inspect: {run_dir.name}: {e}", file=sys.stderr)
            return 1
    else:
        if (state := R.summarize(run_dir)["state"]) not in ("running", "waiting"):
            print(f"research inspect: {run_dir.name} is {state}: no live workload to dump", file=sys.stderr)
            return 1
        results = pyspy.of_run(run_dir)
        if results is None:
            print(f"research inspect: {pyspy.MISSING}", file=sys.stderr)
            return 3
        text = pyspy.render(results)
    sys.stdout.write(text)
    return 0


CLUSTER_SRC = Path(__file__).resolve().parents[4] / "infra" / "cluster" / "src"


def cmd_usage(argv: list[str]) -> int:
    """`research usage ...` is `cluster usage ...`, run from this tree's infra/cluster (research never imports cluster), with
    its output and exit code: 0, 1 on a finding (the record unread, stale or short of hours), 2 on bad arguments or with no
    infra/cluster here."""
    import subprocess

    if not (CLUSTER_SRC / "cluster" / "usage.py").is_file():
        print(f"research usage: no infra/cluster beside this research package ({CLUSTER_SRC})", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CLUSTER_SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "cluster", "usage", *argv], env=env).returncode


OPS_COMMANDS = ("keeper", "deploy")
OPS_SRC = Path(__file__).resolve().parents[4] / "infra" / "ops"


def hand_to_ops(cmd: str, rest: list[str]) -> int:
    """`research keeper|deploy ...` is `ops ...` (infra/ops), exec'd in this interpreter with this tree's
    infra/ops first on its path, so a venv synced before verity-ops existed still runs it. research never imports it."""
    env = dict(os.environ)
    if (OPS_SRC / "verity_ops" / "cli.py").is_file():
        env["PYTHONPATH"] = str(OPS_SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, "-m", "verity_ops", cmd, *rest], env)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "--version":
        from . import __version__

        print(__version__)
        return 0
    from .secret_hint import recover

    recover()
    cmd, rest = argv[0], argv[1:]
    if cmd in OPS_COMMANDS:
        return hand_to_ops(cmd, rest)
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "status":
        return cmd_status(rest)
    if cmd == "inspect":
        return cmd_inspect(rest)
    if cmd == "fetch":
        return cmd_fetch(rest)
    if cmd == "cancel":
        return cmd_cancel(rest)
    if cmd == "compare":
        return cmd_compare(rest)
    if cmd == "launch-request":
        return cmd_launch_request(rest)
    if cmd == "env":
        return cmd_env(rest)
    if cmd == "doctor":
        from .doctor import main as doctor_main

        return doctor_main(rest)
    if cmd == "disk-hold":
        from .disk import main as disk_main

        return disk_main(rest)
    if cmd == "cpu-hold":
        from .cores import main as cores_main

        return cores_main(rest)
    if cmd == "job-exec":
        from .jobuser import main as job_exec_main

        return job_exec_main(rest)
    if cmd == "mem":
        from .mem import main as mem_main

        return mem_main(rest)
    if cmd == "tele":
        from .telemetry.cli import main as tele_main

        return tele_main(rest)
    if cmd == "data":
        from .store.cli import main as data_main

        return data_main(rest)
    if cmd == "keep":
        from .retention import keep_main

        return keep_main(rest)
    if cmd == "retention":
        from .retention import main as retention_main

        return retention_main(rest)
    if cmd == "pods":
        from .pods.part import main as pods_main

        return pods_main(rest)
    if cmd == "monitors":
        from .monitors import main as monitors_main

        return monitors_main(rest)
    if cmd == "notes":
        from .notes import main as notes_main

        return notes_main(rest)
    if cmd == "digest":
        from .digest import main as digest_main

        return digest_main(rest)
    if cmd == "merge":
        from .merge import main as merge_main

        return merge_main(rest)
    if cmd == "queue":
        from .queue import main as queue_main

        return queue_main(rest)
    if cmd == "jobs":
        from .jobs.cli import main as jobs_main

        return jobs_main(rest)
    if cmd == "worker":
        from .jobs.cli import worker_main

        return worker_main(rest)
    if cmd == "git":
        from .commitmap import main as git_main

        return git_main(rest)
    if cmd in ("msg", "slack"):
        from .slack import main as slack_main

        if cmd == "slack":
            print("research: `research slack` is `research msg`; use `research msg`", file=sys.stderr)
        return slack_main(rest, prog="research msg")
    if cmd == "usage":
        return cmd_usage(rest)
    if cmd == "agents":
        from .agents import main as agents_main

        return agents_main(rest)
    age = checkout_age()
    print(f"research: unknown command {cmd!r}" + (f". {age} If {cmd!r} is newer than that, update the checkout "
                                                  "(`git pull`) and retry." if age else "") + f"\n\n{__doc__.strip()}",
          file=sys.stderr)
    return 2


def checkout_age(repo: Path = Path(__file__).resolve().parent) -> str:
    """Where this checkout's HEAD stands: its commit, and how far behind origin/main it is as of the last fetch (no network,
    so a credential prompt can't hang it; no date, since merge trains pin theirs). Empty outside a git checkout."""
    import subprocess

    def git(*a: str) -> str:
        p = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=20,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        return p.stdout.strip() if p.returncode == 0 else ""

    try:
        sha = git("rev-parse", "--short", "HEAD")
        behind = git("rev-list", "--count", "HEAD..origin/main")
    except (OSError, subprocess.SubprocessError):
        return ""
    if not sha:
        return ""
    if not behind:
        return f"This checkout is at {sha}."
    n = int(behind)
    return (f"This checkout is at {sha}, "
            + (f"{n} commit{'s' if n != 1 else ''} behind origin/main as of its last fetch." if n else
               "level with origin/main as of its last fetch."))


if __name__ == "__main__":
    sys.exit(main())
