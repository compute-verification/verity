# recursion_e2e: design

What the runner drives and where, each option of a run, and the pipeline step by step. [`README.md`](README.md) says
what the bench is, gives one thing to try, and says what its verdict and its numbers cover.

## The pipeline it times

Recursive witness zero knowledge proves an inner statement with Flock's circuit prover, ZK off, against the inner phase's ZK server. It then
proves V*, the inner verifier written as Flock statements, with `--zk`, and the auditor's Lean verifier checks V*'s
releases only. This is the one runner for that pipeline. It drives the steps of the pod script,
`verity/protocols/proofs/flock/prover/pod/85-rec-reprice.sh`, as one recorded run, times each step on the smallest real input (K = 4096,
N = 2), repeats what is cheap to repeat, and says whether the run was whole (every step of one recursive proof), whether its parts
passed, and whether it was accepted: Lean composed V*'s parts into the recursive statement and accepted it. Speed work measures each cut
before and after with it.

## A run's options

- **Where it runs.** A run with no GPU prove (behind the ZK server, on the pod script's CPU prover, the default; or `--cpu`) is a plain run on a
  CPU node (`CPU_NODE` in `run.py`): `job.sh` with `GPU=0 FLOCK_WORK=/workspace/cache/rec-bench`, pinned to `--cores`
  (`taskset -c 0-31`), under `--mem-gb` and a timeout of `--max-min`; `run --dry-run` prints the launch. `--on` picks
  another of the CPU nodes that take plain runs (`PLAIN_CPU_NODES`), and `--cores` its cores. The nodes that take runs
  only through their slots (`SLOT_ONLY`) refuse a CPU run. `--on auto` hands the run to `research run`'s queue, which
  places it itself: the queue's options without `--queue`, and for a CPU run no `--gpu-host` and the CPU node's
  `GPU=0 FLOCK_WORK=...`, since it may land on one. A GPU run (`--oprove-gpu 1`, or modes without the ZK server)
  is a GPU host job on a GPU node's queue (`NODE`), and `--preemptible` queues it as a guest on a node that admits guests
  only so.

- **ZK server on by default.** `--modes zk_server`, the pod script's name for the mode through the ZK server: AGENTS.md
  has outer proofs go through the ZK server one at a time.
  `dev` and `host` are there to compare against, never the default. A run never proves two of its V* statements at once:
  the pod script proves 4 at once by default, so `job.sh` sets its `OPROVE_JOBS=1` and records it in `params.txt` (the table's header
  says "V* statements 1 at a time").
- **The lease, per statement.** A V* release at K = 4096 takes minutes, and its time is the Lean session
  loop's, not the prover's. Each statement's lease (`stmt_lease_min` in `run.py`, printed by `plan`) is 1.5 times its
  measured release on the CPU prover, one at a time, end to end (`RELEASE_S` in `run.py`). No one ratio to the
  dev run fits, so each statement has its own. The pod script takes one `GPU_MAX_MIN` for every prove of an invocation, so
  `--lease-min` defaults to the largest lease among the statements `LEVELS` selects.
  `--max-min` defaults to the sum of those statements' leases, plus each other step's minutes per repeat and 5 minutes
  for the job's start and result. The leases are not preemptible (`job.sh`), because a preempted release is a lost
  measurement.
- **Memory.** `--mem-gb` defaults to 128 with an oprove behind the ZK server, else 64. It is the run's memory reservation, and a queued node makes it the run's MemoryMax.
  L0's release needs the most: the Lean session loop and the prover together.
- **The CPU prover under the ZK server** (`--oprove-gpu auto`, which is 0 when oprove runs through the ZK server). A lease pool stops a
  lease whose GPU computes nothing for about 10 minutes, and the GPU computes nothing while the Lean session loop works. So V*'s statements are
  proved by the pod script's CPU prover (`GPU=0`, built once beside the GPU one), and no lease is needed. The release's time is then
  the ZK server's, plus a CPU prove where a GPU prove would take about a second. `--oprove-gpu 1` puts the GPU prover back
  under `--lease-min`.
- **`--e2e`**, the one-pass mode: build, inner, vstage, vcap, oprove, lean and negative once each, one
  run per prove and no warm one. With `--cpu` it is the CPU prover end to end (`GPU=0`, no lease), on a
  CPU node.
- **The session's times** (the result's `session_times`): each step of one recursive session after its build (inner,
  vstage, vcap, oprove, lean), its first invocation's wall and CPU seconds and peak RSS, what the session's figures (its
  cost, knowledge-soundness error and stop-leak bound per window and pooled over the monitoring periods) scale from. The
  figures read the census and the periods: `benchmarks/recursion_figures/figures.py RESULT.json` gives them.
- **Links** (`--env REC_LINKS=1`, default off): the inner population wired. build stages its N units as a chain
  (`rec_reprice.py stage-chain`), each cut wire a link the inner statement's public file carries (META links), which
  `serve`, `replay` and `statement` check against the verifier's links file (`stage/links.json`). vstage stages V*'s link
  table part over it (`rec_vstage stage-links`: `links`, the table registered as V*'s value `rec-links`, and the forgery
  `forged-link`, a table whose first row names a writer whose output is not its reader's row), oprove proves both beside
  the other statements, and `rec-inner`, every verify and `rec-compose` read the table's `--rec-links ROOT:N`. Lean's
  `verify`, `statement` and `rec-compose` read it too (`RecLinkTree.linksArg`): V*'s link part (`RecLinkPart.check`)
  holds every row of the table to the inner statement's, and `rec-compose` requires every part to name the composition's
  table, so the lean step judges the link part and `forged-link` like any other part. `--max-min` budgets the link
  part's releases and its forgery's each as the largest measured statement's (`max_min` in `run.py`).
- **The numbers** (`metrics.toml`): `record.py run` is `run.py run` with `record.py job` around job.sh, which after
  job.sh writes two of the headline's figures into the run's result through the harness kit (`record.py`'s `METRICS`):
  `pipeline_sum_s`, one recursive run as an auditor waits for it, and the Lean step's seconds within it, each measured
  when the run was whole and partial when it wasn't. The run's validation is the verdict: passed only for an accepted
  recursive run.
- **Data.** the pod script's staged statements and runs go under `$FLOCK_WORK/data/rb-<run id>` and are removed at the end unless
  `--env KEEP_DATA=1`. A later run takes them with `--env TAG=rb-<run id> --env KEEP_DATA=1 --steps oprove,...`. Runs on
  one TAG must not overlap oprove and lean, because both use `data/$TAG/v-sess`.
- `--env` passes anything on to `job.sh` and the pod script: `REPEAT_<STEP>=n` sets one step's repeats, plus `LEVELS=L0,cap`,
  `LEAN_JOBS`, `PCPUS`, `WARM`, `RUNS` (the inner's runs) and the like.
- **Settings.** `settings.py` declares every variable `job.sh` reads (and `run.py`'s and `forge_after.py`'s), each with its
  type, default, doc and role (`value`, `where` or `debug`). `REPEAT_` and `OPROVE_` are the job's own prefixes: `run.py`
  refuses an `--env` under one of them that nothing reads (`REPEAT_LAEN=1`), or a declared value that doesn't parse, before
  it launches, and `job.sh`'s first command refuses the same on the node. `job.sh` writes the value settings it was given to
  `out/settings.json`, and `params.txt` (so the result's `params`) carries their SHA-512 digest as `settings`. the pod script's own
  variables pass through undeclared.

- **Another inner statement** (`stage.sh`, launched as `job.sh` is): `rec_inner.py stage --subcircuit` stages N seeded
  inputs to a census-style subcircuit into `data/$TAG/stage` (plain leaves) and `stage-zk` (hm96), or, with
  `WINDOW=DIR`, `rec_inner.py stage-window` stages every proof block of a served window's member from the window's own
  rows and salts (`prove_path.py commit`'s window dir). It then runs `job.sh` with 85's `STAGE=0`, which proves what it
  finds.

## The pipeline (step map)

Every step is a STEP of the pod script (read as it is), run on one node with the big files under
`$FLOCK_WORK/data/$TAG` for the next step. Each GPU step leases one RTX PRO 6000 per prover invocation (`on_gpu`), and
everything else is CPU. The where column is a GPU run's: a CPU run (the default) runs every step on its CPU node
with `GPU=0`. The inner statement is `N` template instances of `GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}`
(`pod/rec_reprice.py`, the population program cut by `Q_template_instance`), with every proof block sampled.

| # | step (STEP) | command (inside the pod script) | reads | writes | where |
|---|---|---|---|---|---|
| 0 | `build` | `60-circuit.sh MODE=build` (the prover), `lake build verity-verify-proof` (Lean), `class_statement --stage-only` twice; `lake build flock-zk-server` (the ZK server's Lean session loop) from the same Lean sources; with `REC_LINKS=1`, `rec_reprice.py stage-chain` instead of `class_statement` | the tree | `stage/` (plain SHA-512 leaves: the recursive inner), `stage-zk/` (hm96 leaves: the non-recursive `--zk`, the baseline), and with `REC_LINKS=1` each one's `links.json` | CPU |
| 1 | `inner` | `flock-circuit prove --gpu` (ZK off, `FC_COINS=tree`: every coin the challenger's coin tree's) against `python -m verity_flock.rec_live --coin-tree` (the inner ZK server's proxy; its trees from the challenger's device at `TREE_FROM`, else sampled by this host's OS, `--tree-test-os`, a test's coins); then the non-recursive ZK-off and `--zk` statements on loopback (`INNER_LOOPBACK=1`); `RELEASE=1`: the recursive session's parts and schedule first (`verity-verify-proof rec-statements`, `rec_inner.py release-parts`), and the session opened on the ZK server's state (`rec_live --zk-server`), no loopback | `stage/`, `stage-zk/` | `inner/s<k>/`, `release.json`, `fw/` | GPU |
| 2 | `icheck` | `flock-circuit replay` of the last inner run (upstream's verifier, a cross-check), `flock-circuit statement` and the schedule each session was held to, and how many of the session's clear tops its public record holds (`tops.txt`); it decides no verdict: V*'s verdict is the Lean verifier's, at `lean` | `inner/` | records only | CPU |
| 3 | `vstage` | `rec_vstage points`, `inner-fold` (`InnerFold.lean` as an executable), `rec_vstage stage` per statement: each Ligerito level's `RecOpen`, the algebra's `InnerRepCheck` parts, and `--forge`'s forged statements (`FORGE`); with `REC_LINKS=1`, `rec_vstage stage-links` (the link table's part and `forged-link`) | `inner/`, `stage/` | `v/L<l>/`, `v/alg/p<j>/`, `v/forged-<kind>/`, `v/links/`, `v/forged-link/links/` | CPU |
| 3b | `vcap` | `rec_vstage.stage_cap`: `RecCap` over each level's committed tops (V*'s verdict needs them) | `inner/`, `v/` | `v/cap/`, `v/cap/L<l>/` | CPU |
| 4 | `oprove` | V*'s own inputs fixed before any session (`verity-verify-proof rec-inner`, `rec-statements`, the shared rows, each statement's flags and bindings); per V* statement, one at a time: `flock-circuit serve --zk` (loopback) and `prove --zk --gpu` in each mode (`zk_server`: `FC_ZK_SERVER=1`, the Lean session loop `flock-zk-server`, live PROTOCOL.md §10.4, and the run's record `v-sess/rec-run.jsonl`: staged first, each Hello logged by `serve --rec-run`, closed last); `RELEASE=1`: each part once, in order, through `prove --zk-server-state` (`core-zk-server session` in front of the loop), then `rec_inner.py release-report` | `v/`, `inner/`, `fw/` | `v-sess/`, `fw/` | GPU |
| 5 | `oreplay` | `flock-circuit replay --zk` of each V* release (upstream's verifier) | `v-sess/` | records only | CPU |
| 6 | `lean` | `verity-verify-proof verify --zk --rec-part J` on each V* part's release with the V* inputs oprove fixed (`--rec-inner-*`, `--rec-rows`, `--rec-bindings`), `verity-verify-proof statement` on the inner statement (not Lean's plain verify of the inner run itself, whose `--coins tree` is a tree of the statement's own streams, not the inner session's); `verity-verify-proof rec-compose` over the run's record composes the parts' verdicts into V*'s, which the Lean verifier alone decides; `lean_verdicts.py` fails closed | `v-sess/`, `stage/`, `inner/` | records only | CPU |
| 6b | `forge-after` | `forge_after.py`: coef, dirs and message forged after proving on copies of the honest L0 and alg-p0 statements, each the Lean verifier's `verify --zk` with step 6's flags beside its honest control (not a STEP of the pod script) | `v-sess/`, `v/`, step 6's `own-<stmt>.txt` (step 4's) | records only | CPU |
| 7 | `negative` | step 6 with `NEG_INNER`: each V* release again over each flipped copy of the inner run | as 6 | records only | CPU |
