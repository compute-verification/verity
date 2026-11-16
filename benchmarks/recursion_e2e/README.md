# recursion_e2e

Recursion replaces the proofs an auditor checks with one outer proof that the verifier accepted them. The developer
proves an inner statement with Flock, zero knowledge off, then proves V*, the inner verifier written as Flock
statements, with zero knowledge, and the auditor's Lean verifier checks V*'s releases only. This bench runs that
pipeline end to end on the smallest real input, as one recorded run, and times each step. It says whether the run was
whole, whether its parts passed, and whether Lean composed them into the recursive statement and accepted it.

This pipeline hides the inner witness over a public circuit. The run that also hides the circuit, recursion with
`InnerFold.lean --hide` over a hidden RoPE proof block, is `verity/protocols/proofs/flock/prover/pod/88-rec-c3-inner.sh`; this runner doesn't
time it yet.

## Try it

~~~sh
uv run verity bench recursion_e2e plan
~~~

This prints the steps a run would take, each statement's lease in minutes, and the `research run` command that would
launch it, and starts nothing. It needs no GPU. `verity bench recursion_e2e run` launches that command.

~~~text
verity bench recursion_e2e plan  [POINT]                          the steps, each statement's lease, the launch command
verity bench recursion_e2e run   [POINT] [--e2e] [--release] [--dry-run] [--here]
                                                                   one recorded run of job.sh (a run behind the ZK server
                                                                   or a --cpu run: a plain run on a CPU node; a GPU run: a
                                                                   GPU node's queue)
verity bench recursion_e2e result --dir OUT [--out F] [--md F]    the result from a job's out/ (the job runs it last)
verity bench recursion_e2e table RESULT... [--merge] [--json]     the table from results (a file, or a run id whose
                                                                   records `research fetch --all RUN` fetched, read with
                                                                   this reader)
verity bench recursion_e2e store RESULT... [--merge] --dir DIR    the results and their table as one art's tree in DIR,
                                                                   and the `research data put` command that stores it
POINT: --K 4096 --N 2 --steps S,.. --repeat 3 --modes zk_server --oprove-runs 1 --lease-min M --neg challenge,query
       --forge KINDS [--no-forge-after] --cpu --on MACHINE --cores 0-31 [--preemptible] --mem-gb G --cpus 32 --max-min M --env KEY=VALUE ...
~~~

[`DESIGN.md`](DESIGN.md) says where a run goes, what each option does, and what each step of the pipeline runs.

## What the verdict says

- **The negative** (`negative`, on by default): the lean step again over a copy of the inner run with one coin of
  rep 0 flipped (the pod script's `NEG_INNER`, `rec_inner.py forge`), once per `--neg` coin. A flipped coin moves only the V*
  statements that read it, so a coin counts as refused when Lean refuses it under at least one statement (exit 1, no
  VERDICT accepting) and every statement has a verdict over it. This is `lean_verdicts.py`'s rule. The row names the
  statements that refused it and those that rightly accepted it.
- **Forgeries** (`--forge`, default none): the pod script's forged statements (`FORGE`, `rec_vstage --forge`: message, sum, comb,
  top, top-salt, coef, dirs, filler), staged by vstage and proved by oprove one run each. A forgery is refused when
  one of its statements is, whether the ZK server stopped it before release (`ZK-SERVER-REFUSED`, an older run's
  `FIREWALL-REFUSED`) or Lean refused it, and a statement it leaves as the honest one is accepted (forged-message's alg-p1). This is `lean_verdicts.py`'s rule. Its
  rows are left out of the outer totals. `--max-min` counts every statement a forgery stages (`FORGED`: message, comb
  and filler every algebra part, sum those and L0 and L3), each sized as its honest statement's release. With coef, dirs
  or message among them, the **`forge-after`** step runs after lean (unless
  `--no-forge-after`; `--steps ...,forge-after` asks for it alone). The ZK server stops those three before Lean sees them,
  so `forge_after.py` makes each one after proving instead, on copies of the honest L0 and alg-p0 statements: row 0 of
  the port one bit off, as its `b || c` alone, as `pub.bin`'s consistent table, and as the whole registration. Lean
  verifies each with the lean step's own flags, beside the honest control, which must accept. The headline's `forged_refused` gives
  each kind's verdict.
- **`--release`**, the recursive session through the ZK server's release side (live PROTOCOL.md §10.5; the pod script's
  `RELEASE=1`), one pass. The inner step fixes the session's parts (V*'s statements at the inner statement's shape, in
  `rec-statements`' order) and its release schedule (`RELEASE_WINDOWS_S`: a window per part by its class, and a lead before
  the first; `rec_inner.py release-parts`) before its one inner session, which opens the recursive session on the ZK
  server's state (`rec_live --zk-server`) on a stream whose failure budget is `RELEASE_BUDGET` (1 unless `--env
  RELEASE_BUDGET=b`: the ZK server opens no recursive session on a stream without one). oprove then proves each part once,
  in order, through
  `flock-circuit prove --zk --zk-server-state DIR --session ID --part NAME`: the ZK server's Lean session loop behind
  the release side's own Lean loop (`core-zk-server session`), which holds each verdict to its release time and stops the
  session at a part that is late or refused. `rec_inner.py release-report` reads the ZK server's record against the
  schedule and the Lean loop's `FIREWALL-TIMES` lines (when it decided each part) into `out/oprove/release.json`; the step
  fails unless every part was released at its time. With `--env TREE_FROM=local` the challenger runs beside the inner
  step, every inner coin is its coin tree's, and `rec_inner.py tree-held` holds the inner session to its signed release log
  (`out/inner/tree-held.json`, by `rec_live.tree_held_reason`, the Python mirror of the check the recursive audit reads,
  `Spot.Challenger.innerTreesReason`). The result's `release` and the table's release-side section give each
  part's window, release time, decision time, slack and prove wall; the headline's `released` is its verdict, and `ok`
  needs it. `--max-min` gives oprove twice its proving model (a window is about twice its part's prove) and the lead.
- **The verdict** (`headline` in the result, the table's first line): `complete` when every step that one recursive
  run needs ran (build, inner, vstage, vcap, oprove, lean). `parts_ok` when, in addition, every invocation exited 0, every
  honest verdict accepted, every flipped coin and every forgery was refused. `composed` when Lean composed V*'s parts
  into the recursive statement, which the lean step records beside its logs (`lean/rec-compose.json`): it composes only
  a run of every V* part whose challenger kept its record of the run, as the default modes do with one session a
  statement behind the ZK server, so a `dev` or `host` run, or one of some of `LEVELS`, exits 0 on its parts and is not
  composed. `ok`, an accepted recursive run, only when its parts passed and Lean accepted the composition. For a run
  that passed on its parts and was not composed, the table's first line says so, with the oprove modes and the reason,
  and the `compose` row has no verdict. A run whose composition Lean refused is not accepted, and the line gives
  Lean's reason (`refused`): a forced retry's (`RETRY`) starts with the retry. A merged table's verdict is only as
  whole as its runs.

What the recursive audit has that the pod script doesn't run, so no number here covers it:

- **One verdict over the whole recursive run.** STEP=lean verifies each V* release on its own and composes their
  verdicts once (`rec-compose`, over the run's record), but no one command takes the inner record and every V* release
  and answers once.

## What each number covers

`job.sh` runs STEP `build` once, then each step of `--steps` in order. Every CPU step runs `--repeat` times (default 3).
Each GPU step runs once, with its runs inside it: the inner's `WARM=1 RUNS=3`, each V* statement's
`--oprove-warm 0 --oprove-runs 1`. Each invocation of the pod script runs under its own `RESEARCH_RUN_DIR`
(`out/steps/<step>/r<i>/`), so every record it writes is kept. `result` reads them into `out/recursion_e2e.json` and
`out/table.md`. `job.sh` tries to write them on SIGTERM too, but cancelling the run ends its scope within a second,
which leaves only the records. `table RUN` rebuilds the result from them once `research fetch --all RUN` has fetched
them, so a cancelled run's rows are still read.

- **Prover time** (`inner`, `oprove`; the table's "time s"): each timed run's `prove_total_s` from the prover's `LIVE`
  line (both reps' `prove_s` summed). This covers the device's phases, not the host witness.
- **Run e2e**: each timed run's `e2e_s`, from the run's start to its verdict. It includes the host witness
  and every wait on the live verifier. Median and [min, max] are over the timed runs, and warm runs are left out.
- **Pace** (`session_s`, as the pod script's summary computes it): the time from the last warm run's verdict to the last timed
  one's, divided by the timed runs. It needs a warm run.
- **Dominant phase**: the largest part of the median run's `e2e_s`. The candidates are the prover's own phase buckets
  (`t.*`), the host witness (`h.witness`) and `wait.verifier`, which is what is left of `e2e_s` (the run waiting on
  its verifier: the `rec_live` proxy, or `serve`, whose `verify_s` and `handle_s` are in the row's `extra`). Behind
  the ZK server, the prover's buckets also hold its waits on the Lean session loop, and the two reps overlap. So there the
  cost is read from the loop's own process: `flock-zk-server`'s (`flock-firewall`'s in a recorded run) CPU seconds, peak RSS and lifetime, from the run's process
  samples (`resources.jsonl`, the run's telemetry beside `out/`).
- **Peak memory**: the prover process's peak RSS (`/usr/bin/time -v` over the invocation) and the leased GPU's memory in
  use, sampled every half second (`gpu-mib.txt`, a lower bound). For CPU steps, each tool's `/usr/bin/time -v` peak RSS.
- **CPU steps** (`icheck`, `vstage`, `vcap`, `oreplay`, `lean`, `negative`): each tool's `/usr/bin/time -v` wall, median
  and [min, max] over the repeats. Lean's verify also splits into `inputs`, `setup` and the verification's own checks. The pod script
  runs `vstage`'s statements and `lean`'s verifications `JOBS` at a time (4 here), so each one's wall is measured under its
  neighbours' load. The step's whole wall (`lean step`) is what an auditor waits, and it is what the pipeline sum adds.
  `stage <name>` rows carry what `rec_vstage` printed of each proof block (template, rows, lowering seconds, circuit bytes).
- **Sizes**: the V* proofs' bytes (`proof_bytes`, what the auditor receives), the inner proof's (the prover's witness,
  never sent), the transcript bytes each way, and the staged files (`sizes.txt`).
- **The pipeline sum**: the inner run's e2e, the vstage and vcap steps' walls, the outer prove's run e2e summed over
  every statement in the mode that proved the most of them, and the lean step's wall. It is one recursive run's
  path, as long as an auditor waits for it.
- **Setup** (`build`): the step's wall and peak, once, cold or cached as the job found it.

What the numbers leave out:

- without `--release`, the ZK server's release side: each V* statement is proved against its own `serve`, with no schedule;
- network latency: every verifier is on loopback on the same node, on the job's other cores;
- queue and lease waits: the lease's own lines are kept but not counted;
- contention: the node's other jobs share its cores. A run's `prover_cores_busy_before` says how loaded its cores were,
  and two runs side by side (each its own TAG) share the node's memory bandwidth.
