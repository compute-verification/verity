# recursion_e2e: the recursive session, timed stage by stage (`verity bench recursion_e2e`)

Recursive witness zero knowledge proves an inner statement with ZK off (M0) against the inner phase's firewall. It then
proves V*, the inner verifier written as C-Flock statements, with `--zk`, and the auditor's Lean verifier checks V*'s
sessions only. This is the one runner for that pipeline. It drives `verity/ml/flock/pod/85-rec-reprice.sh`'s steps as one
research run, times each stage on the smallest real input (K = 4096, N = 2), repeats what is cheap to repeat, and says
whether the run was one whole recursive session and whether it was accepted. Speed work measures each cut before and after
with it.

~~~text
verity bench recursion_e2e plan  [POINT]                          the stages, each statement's lease, the launch command
verity bench recursion_e2e run   [POINT] [--e2e] [--dry-run] [--here]
                                                                   one research run of job.sh (a firewalled or --cpu run: a
                                                                   plain run on vy-nebius-cpu-2; a GPU run: vy-nebius-1's queue)
verity bench recursion_e2e result --dir OUT [--out F] [--md F]    the result from a job's out/ (the job runs it last)
verity bench recursion_e2e table RESULT... [--merge] [--json]     the table from results (a file, or a run id whose
                                                                   records are fetched and read with this reader)
verity bench recursion_e2e store RESULT... [--merge]              the results and their table as one art: id
POINT: --K 4096 --N 2 --steps S,.. --repeat 3 --modes firewall --oprove-runs 1 --lease-min M --neg challenge,query
       --forge KINDS [--no-forge-after] --cpu --on MACHINE --cores 0-31 [--preemptible] --mem-gb G --cpus 32 --max-min M --env KEY=VALUE ...
~~~

- **Where it runs.** A run with no GPU prove (firewalled, on 85's CPU prover, the default; or `--cpu`) is a plain run on a
  CPU node: `research run --on vy-nebius-cpu-2 ... --mem-gb 128 --timeout <max-min>m -- taskset -c 0-31 bash job.sh ...
  GPU=0 FLOCK_WORK=/workspace/cache/rec-bench` (infra, 24 Oct). `--on` picks another of vy-nebius-cpu-2 to cpu-5, and
  `--cores` its cores. vy-nebius-1, 2 and 3 and vy-nebius-cpu-1 take runs only through their slots, so a CPU run is refused
  there. A GPU run (`--oprove-gpu 1`, or modes without the firewall) is a GPU host job on vy-nebius-1's queue, and
  `--preemptible` is for a guest on another lane's node (vy-nebius-2 admits guests only so).

- **Firewall on by default.** `--modes firewall`: AGENTS.md has outer proofs go through the firewall one at a time.
  `dev` and `host` are there to compare against, never the default. A run never proves two of its V* statements at once:
  85 proves 4 at once by default, so `job.sh` sets its `OPROVE_JOBS=1` and records it in `params.txt` (the table's header
  says "V* statements 1 at a time"). r20261025-094601-4794 ran before that and timed its sessions 4 at once.
- **The lease, per statement.** A firewalled V* session at K = 4096 takes minutes, and its time is the Lean session
  loop's, not the prover's. Each statement's lease (`stmt_lease_min` in `run.py`, printed by `plan`) is 1.5 times its
  measured firewalled session (r20261024-094937-6572, the CPU prover, one at a time, end to end): L0 3738 s, L1 894,
  L2 836, L3 469, alg-p0 815, alg-p1 847, the caps 51–86. No one ratio to the dev session fits: L0 took 626 times its
  dev session, L1–L3 190–315 times, alg-p0 62 times. 85 takes one `GPU_MAX_MIN` for every prove of an invocation, so
  `--lease-min` defaults to the largest lease among the statements `LEVELS` selects (94 min with L0, 12 with `LEVELS=L3`).
  `--max-min` defaults to the sum of those statements' leases, plus each other step's minutes per repeat and 5 minutes
  for the job's start and result. The leases are not preemptible (`job.sh`), because a preempted session is a lost
  measurement.
- **Memory.** `--mem-gb` defaults to 128 with a firewalled oprove, else 64. It is the run's memory reservation, and a queued node makes it the run's MemoryMax,
  and L0's firewalled session peaked at 99 GB: 57 GB in the Lean session loop and 31 GB in the prover.
- **The CPU prover under the firewall** (`--oprove-gpu auto`, which is 0 with `--modes firewall`). Node 1's lease pool
  stopped every firewalled L0 session at about 10 minutes (exit 143) despite a 74-minute, non-preemptible lease. The
  run was r20261024-075309-69be, and the GPU computes nothing while the Lean session loop works. So V*'s statements are
  proved by 85's CPU prover (`GPU=0`, built once beside the GPU one), and no lease is needed. The session's time is then
  the firewall's, plus a CPU prove where a GPU prove would take about a second. `--oprove-gpu 1` puts the GPU prover back
  under `--lease-min`.
- **`--e2e`**, the one-pass mode (RECBENCH's draft): build, inner, vstage, vcap, oprove, lean and negative once each, one
  session per prove and no warm one. With `--cpu` it is the CPU prover end to end (`GPU=0`, no lease), on a
  CPU node.
- **The negative** (`negative`, on by default): the lean step again over a copy of the inner session with one coin of
  rep 0 flipped (85's `NEG_INNER`, `rec_inner.py forge`), once per `--neg` coin. A flipped coin moves only the V*
  statements that read it, so a coin counts as refused when Lean refuses it under at least one statement (exit 1, no
  VERDICT accepting) and every statement has a verdict over it. This is `lean_verdicts.py`'s rule. The row names the
  statements that refused it and those that rightly accepted it.
- **Forgeries** (`--forge`, default none): 85's forged statements (`FORGE`, `rec_vstage --forge`: message, sum, comb,
  top, top-salt, coef, dirs, filler), staged by vstage and proved by oprove one session each. A forgery is refused when
  one of its statements is, whether the firewall stopped it before release (`FIREWALL-REFUSED`) or Lean refused it, and
  a statement it leaves as the honest one is accepted (forged-message's alg-p1). This is `lean_verdicts.py`'s rule. Its
  rows are left out of the outer totals. `--max-min` sizes each forged session as its honest statement's
  (`FORGED_AS`). With coef, dirs or message among them, the **`forge-after`** step runs after lean (unless
  `--no-forge-after`; `--steps ...,forge-after` asks for it alone). The firewall stops those three before Lean sees them,
  so `forge_after.py` makes each one after proving instead, on copies of the honest L0 and alg-p0 statements: row 0 of
  the port one bit off, as its `b || c` alone, as `pub.bin`'s consistent table, and as the whole registration. Lean
  verifies each with the lean step's own flags, beside the honest control, which must accept (first run
  r20261024-160722-958b: all refused at setup, the controls accepted; as a step, r20261024-172707-4385 at K = 64, N = 16
  with `--forge coef,dirs,message`: complete and accepted, every forgery refused). The headline's `forged_refused` gives
  each kind's verdict.
- **The verdict** (`headline` in the result, the table's first line): `complete` when every step that one recursive
  session needs ran (build, inner, vstage, vcap, oprove, lean). `ok` when, in addition, every invocation exited 0, every
  honest verdict accepted, every flipped coin and every forgery was refused. A merged table's verdict is only as whole as its runs.
- **Data.** 85's staged statements and sessions go under `$FLOCK_WORK/data/rb-<run id>` and are removed at the end unless
  `--env KEEP_DATA=1`. A later run takes them with `--env TAG=rb-<run id> --env KEEP_DATA=1 --steps oprove,...`. Runs on
  one TAG must not overlap oprove and lean, because both use `data/$TAG/v-sess`.
- `--env` passes anything on to `job.sh` and 85: `REPEAT_<STEP>=n` sets one step's repeats, plus `LEVELS=L0,cap`,
  `LEAN_JOBS`, `PCPUS`, `WARM`, `RUNS` (the inner's sessions) and the like.

## The pipeline (stage map)

Every stage is a STEP of 85 (proofs' job, read as it is), run on one node with the big files under
`$FLOCK_WORK/data/$TAG` for the next step. Each GPU step leases one RTX PRO 6000 per prover invocation (`on_gpu`), and
everything else is CPU. The node column is a GPU run's: a CPU run (the default) runs every step on its CPU node
with `GPU=0`. The inner statement is `N` instances of `GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}`
(`pod/rec_reprice.py`, the population program cut by `Q_template_instance`), with every unit drawn.

| # | stage (STEP) | command (inside 85) | reads | writes | node |
|---|---|---|---|---|---|
| 0 | `build` | `60-circuit.sh MODE=build` (the prover), `lake build flock-verify` (Lean), `class_statement --stage-only` twice; `lake build flock-firewall` (the firewall's Lean session loop) from the same Lean sources | the tree | `stage/` (plain SHA-512 leaves: the recursive inner), `stage-zk/` (hm96 leaves: today's `--zk`, the baseline) | n1, CPU |
| 1 | `inner` | `flock-circuit prove --gpu` (M0, ZK off, `FC_COINS=os`) against `python -m verity_flock.rec_live` (the inner firewall's proxy); then today's M0 and `--zk` on loopback (`INNER_LOOPBACK=1`) | `stage/`, `stage-zk/` | `inner/s<k>/` | n1, GPU |
| 2 | `icheck` | `flock-circuit replay` of the last inner session, `python -m verity_flock.rec_vstar` (V*'s hashing reference) | `inner/` | records only | n1, CPU |
| 3 | `vstage` | `rec_vstage points`, `inner-fold` (`InnerFold.lean` as an executable), `rec_vstage stage` per statement: each Ligerito level's `RecOpen`, the algebra's `InnerRepCheck` parts, and `--forge`'s forged statements (`FORGE`) | `inner/`, `stage/` | `v/L<l>/`, `v/alg/p<j>/`, `v/forged-<kind>/` | n1, CPU |
| 3b | `vcap` | `rec_vstage.stage_cap`: `RecCap` over each level's committed tops (V*'s verdict needs them) | `inner/`, `v/` | `v/cap/`, `v/cap/L<l>/` | n1, CPU |
| 4 | `oprove` | per V* statement, one at a time: `flock-circuit serve --zk` (loopback) and `prove --zk --gpu` in each mode (`firewall`: `FC_FIREWALL=1`, the Lean session loop `flock-firewall`, live PROTOCOL.md §10.4) | `v/` | `v-sess/<stmt>/` | n1, GPU |
| 5 | `oreplay` | `flock-circuit replay --zk` of each V* session (upstream's verifier) | `v-sess/` | records only | n1, CPU |
| 6 | `lean` | `flock-verify verify --zk` on each V* session with V*'s own inputs (`--rec-inner-*`, `--rec-rows`, `--rec-bindings`), the inner session (`INNER_LEAN=1`), `flock-verify statement` on the inner statement; `lean_verdicts.py` fails closed | `v-sess/`, `stage/`, `inner/` | records only | n1, CPU |
| 6b | `forge-after` | `forge_after.py`: coef, dirs and message forged after proving on copies of the honest L0 and alg-p0 statements, each `flock-verify verify --zk` with step 6's flags beside its honest control (not a STEP of 85) | `v-sess/`, `v/`, step 6's `own-<stmt>.txt` | records only | n1, CPU |
| 7 | `negative` | step 6 with `NEG_INNER`: each V* session again over each flipped copy of the inner session | as 6 | records only | n1, CPU |

What the recursive audit has that 85 doesn't run, so no number here covers it:

- **The recursive session through the firewall's release side** (live PROTOCOL.md §10.5): `rec_live --firewall DIR --parts
  L0,…,alg` opening one recursive session per inner session, each outer session through `python -m verity_flock.firewall
  relay` or `flock-circuit prove --firewall-state DIR --session ID --part NAME`, one outer slot, statements in order. Only
  the tests run it (`verity/ml/flock/tests/test_rec_live_firewall.py`). 85 proves each V* statement against its own `serve`.
- **One verdict over the whole recursive session.** STEP=lean verifies each V* session on its own, and the runner's
  `ok` is the conjunction of those verdicts. No one command takes the inner record and every V* session and answers once.

## What each number covers

`job.sh` runs STEP `build` once, then each step of `--steps` in order. Every CPU step runs `--repeat` times (default 3).
Each GPU step runs once, with its sessions inside it: the inner's `WARM=1 RUNS=3`, each V* statement's
`--oprove-warm 0 --oprove-runs 1`. Each invocation of 85 runs under its own `RESEARCH_RUN_DIR`
(`out/steps/<step>/r<i>/`), so every record 85 writes is kept. `result` reads them into `out/recursion_e2e.json` and
`out/table.md`. `job.sh` tries to write them on SIGTERM too, but `research cancel` ends the run's scope within a second,
which leaves only the records. `table RUN` rebuilds the result from a fresh fetch of them, so a cancelled run's rows
are still read.

- **Prover time** (`inner`, `oprove`; the table's "time s"): each timed session's `prove_total_s` from the prover's `LIVE`
  line (both reps' `prove_s` summed). This covers the device's phases, not the host witness.
- **Session e2e**: each timed session's `e2e_s`, from the session's start to its verdict. It includes the host witness
  and every wait on the live verifier. Median and [min, max] are over the timed sessions, and warm sessions are left out.
- **Pace** (`session_s`, as 85's summary computes it): the time from the last warm session's verdict to the last timed
  one's, divided by the timed sessions. It needs a warm session.
- **Dominant phase**: the largest part of the median session's `e2e_s`. The candidates are the prover's own phase buckets
  (`t.*`), the host witness (`h.witness`) and `wait.verifier`, which is what is left of `e2e_s` (the session waiting on
  its verifier: the `rec_live` proxy, or `serve`, whose `verify_s` and `handle_s` are in the row's `extra`). Under
  `firewall`, the prover's buckets also hold its waits on the Lean session loop, and the two reps overlap. So there the
  cost is read from the loop's own process: `flock-firewall`'s CPU seconds, peak RSS and lifetime, from the run's process
  samples (`resources.jsonl`, `research run`'s telemetry beside `out/`).
- **Peak memory**: the prover process's peak RSS (`/usr/bin/time -v` over the invocation) and the leased GPU's memory in
  use, sampled every half second (`gpu-mib.txt`, a lower bound). For CPU steps, each tool's `/usr/bin/time -v` peak RSS.
- **CPU steps** (`icheck`, `vstage`, `vcap`, `oreplay`, `lean`, `negative`): each tool's `/usr/bin/time -v` wall, median
  and [min, max] over the repeats. Lean's verify also splits into `inputs`, `setup` and the session's own checks. 85
  runs `vstage`'s stages and `lean`'s sessions `JOBS` at a time (4 here), so each one's wall is measured under its
  neighbours' load. The step's whole wall (`lean step`) is what an auditor waits, and it is what the pipeline sum adds.
  `stage <name>` rows carry what `rec_vstage` printed of each unit (template, rows, lowering seconds, circuit bytes).
- **Sizes**: the V* proofs' bytes (`proof_bytes`, what the auditor receives), the inner proof's (the prover's witness,
  never sent), the transcript bytes each way, and the staged files (`sizes.txt`).
- **The pipeline sum**: the inner M0's e2e, the vstage and vcap steps' walls, the outer prove's session e2e summed over
  every statement in the mode that proved the most of them, and the lean step's wall. It is one recursive session's
  path, as long as an auditor waits for it.
- **Setup** (`build`): the step's wall and peak, once, cold or cached as the job found it.

What the numbers leave out:

- the firewall's release side (above);
- network latency: every verifier is on loopback on the same node, on the job's other cores;
- queue and lease waits: the lease's own lines are kept but not counted;
- contention: node 1's other jobs share cores. A session's `prover_cores_busy_before` says how loaded its cores were,
  and two runs side by side (each its own TAG) share the node's memory bandwidth.
