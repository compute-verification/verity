# recursion: the recursive audit's stages, profiled one by one

Recursive witness zero knowledge proves an inner statement with ZK off (M0) against the inner phase's firewall, then proves
V*, the inner verifier written as C-Flock statements, with `--zk`; the auditor's Lean verifier checks V*'s sessions only.
This benchmark times each stage of that pipeline as it runs on `main` today, on the smallest real input, so the speed work
has numbers before the whole recursive session runs end to end.

~~~text
verity bench recursion profile plan  [--K 4096] [--N 2] [--steps S,..]         the stages, their commands and the launch
verity bench recursion profile run   [--K 4096] [--N 2] [--steps S,..] [--repeat 3] [--modes M,..] [--env K=V ..] [--dry-run]
                                                                                one recorded job on vy-nebius-1
verity bench recursion profile result --dir OUT [--out FILE]                    the profile's result from a job's out/
verity bench recursion profile table RESULT... [--merge] [--json] [--put]       the table, from results (a file or a run id;
                                                                                --merge: one table, a later run's row replacing
                                                                                an earlier one's for the same stage and mode)
~~~

## The pipeline today (stage map)

Every stage is a STEP of `verity/ml/flock/pod/85-rec-reprice.sh` (proofs' job, read as it is), run on vy-nebius-1 with
the big files under `$FLOCK_WORK/data/$TAG` for the next step. Each GPU step leases one RTX PRO 6000 per prover invocation
(`on_gpu`); everything else is CPU. The inner statement is `N` instances of `GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}`
(`pod/rec_reprice.py`, the population program cut by `Q_template_instance`), every unit drawn.

| # | stage (STEP) | command (inside 85) | reads | writes | node | runs today |
|---|---|---|---|---|---|---|
| 0 | `build` | `60-circuit.sh MODE=build GPU=1 SM=120` (the CUDA prover), `lake build flock-verify` (Lean), `class_statement --program-module rec_reprice.py:program --stage-only` twice | the tree | `stage/` (plain SHA-512 leaves: the recursive inner), `stage-zk/` (hm96 leaves: today's `--zk`, the baseline) | n1, CPU | last recorded runs: rec-ksweep (6 Oct, m = 35), the F1 gate (10 Oct, K = 4096, N = 2: `r20261010-210008-003d`) |
| 1 | `inner` | `flock-circuit prove --gpu` (M0, ZK off, `FC_COINS=os`) against `python -m verity_flock.rec_live` (the inner firewall: salted tops, the statement's schedule, `Register` answered with the full draw); then today's M0 and `--zk` on loopback (`INNER_LOOPBACK=1`) | `stage/`, `stage-zk/` | `inner/s<k>/` (`public.json`, `private.json`, `session/`) | n1, GPU | as 0 |
| 2 | `icheck` | `flock-circuit replay` of the last inner session, `python -m verity_flock.rec_vstar` (V*'s hashing reference), the schedule and tree-top controls | `inner/` | records only | n1, CPU | as 0 |
| 3 | `vstage` | `rec_vstage points`, `inner-fold` (`verity/ml/flock/tests/lean/InnerFold.lean` as an executable), `rec_vstage stage` per statement: each Ligerito level's `RecOpen`, the algebra's `InnerRepCheck` parts, the forgeries | `inner/`, `stage/` | `v/L<l>/`, `v/alg/p<j>/`, `v/forged-*/` | n1, CPU | as 0 |
| 3b | `vcap` | `rec_vstage.stage_cap`: `RecCap` over each level's committed tops | `inner/`, `v/` | `v/cap/`, `v/cap/L<l>/` | n1, CPU | `r20261020-002938-635a` (20 Oct) |
| 4 | `oprove` | per V* statement: `flock-circuit serve --zk` (loopback) and `prove --zk --gpu`, per mode: `dev` (the device's witness), `host` (`FC_HOST_WITNESS=1`), `firewall` (`FC_FIREWALL=1`: the Lean session loop, live PROTOCOL.md §10.4) | `v/` | `v-sess/<stmt>/` | n1, GPU | `dev` as 0; `firewall` (24 Oct, `r20261024-054739-343d`): no V* session finishes inside 85's 10-minute lease at K = 4096 (the Lean session loop, `flock-firewall`, holds the prover: 1300 CPU s in 7 min on the prover's 16 cores) |
| 5 | `oreplay` | `flock-circuit replay --zk` of each V* session (upstream's verifier) | `v-sess/` | records only | n1, CPU | as 0 |
| 6 | `lean` | `flock-verify rec-inner`, `rec-statements`, then `flock-verify verify --zk` on each V* session with V*'s own inputs (`--rec-inner-*`, `--rec-rows`, `--rec-bindings`); `INNER_LEAN=1` also the inner session (`--coins os --stratified N`); `flock-verify statement` on the inner statement; `lean_verdicts.py` fails closed | `v-sess/`, `stage/`, `inner/` | records only | n1, CPU | as 0 |

What the recursive audit has that 85 doesn't run, and where it lives today:

- **The recursive session through the firewall's release side** (live PROTOCOL.md §10.5): `rec_live --firewall DIR --parts
  L0,…,alg` opening one recursive session per inner session, each outer session through `python -m verity_flock.firewall
  relay` or `flock-circuit prove --firewall-state DIR --session ID --part NAME` (`core-firewall session`), one outer slot,
  statements in order. Tests only (`verity/ml/flock/tests/test_rec_live_firewall.py`, the record vectors); 85 proves each
  V* statement against its own `serve`.
- **One verdict over the whole recursive session.** STEP=lean verifies each V* session on its own; no command takes the
  inner record and every V* session and answers once.
- **Hidden unit circuits (F4 candidate 3)**: `88-rec-c3-inner.sh` (the inner through `rec_session`'s relay with the
  sparse phase and the stacked table R, `RoPE d64`, CPU) and `87-rec-c3-open.sh` (the opening-only prover, a pytest).
  CPU only, through a check slot; not part of 85.
- **rec-v0's first jobs** (`80-rec-inner.sh`, `81-rec-outer.sh`, `83-rec-alg.sh`, RoPE d64 on the CPU): superseded by 85;
  81's and 83's default inner statement, `verity/flock-circuit`, is refused at setup (`RecSetup.checkForm`).

The e2e run (proofs' lane) is to replace stages 4 to 6's loopback with the firewall's recursive session; this profile's
`profile` family is the pieces, and an `e2e` family is to read that run's records once it exists.

## What each number covers

`profile run` launches one job (`job.sh`): STEP `build` once, then each step of `--steps` in order, every CPU step
`--repeat` times (default 3) and every GPU step once with `WARM=1 RUNS=3` sessions inside it. Each invocation of 85 runs
under its own `RESEARCH_RUN_DIR` (`out/steps/<step>/r<i>/`), so every record 85 writes is kept. `profile result` reads them:

- **Prover time** (`inner`, `oprove`; the table's "time s"): each timed session's `prove_total_s` from the prover's
  `LIVE` line (both reps' `prove_s` summed: the device's phases, not the host witness). **Session e2e**: its `e2e_s`, from
  the session's start to its verdict, the host witness and every wait on the live verifier (coins, `Finish`, the verdict)
  included. The warm session is left out. Median and [min, max] over the timed sessions.
- **Pace** (`session_s`, as 85's summary computes it): the time from the last warm session's verdict to the last timed
  one's, over the timed sessions. Up to `FC_VERIFY_AHEAD` sessions overlap (one proving, the others waiting on verdicts),
  so pace, not e2e, is the stage's throughput, and e2e is one session's latency.
- **Dominant phase**: the largest part of the median session's `e2e_s`, among the prover's own phase buckets (`t.*` of both
  reps' buckets, summed: `t.witness`, `t.encoding_commitment`, `t.zerocheck`, `t.lincheck`, `t.ring_switch`, `t.ligerito`,
  `t.link`, and under the firewall `t.firewall_recheck`, `t.lean_finish`, `t.zk_inner`), the host witness (`h.witness`,
  `witness_s`) and `wait.verifier`, what is left of `e2e_s` (the session waiting on its verifier: the `rec_live` proxy, or
  `serve`, whose own `verify_s` and `handle_s` per session are in the row's `extra`).
- **Peak memory**: the prover process's peak RSS (`/usr/bin/time -v` over the invocation, every session) and the leased GPU's
  memory in use, sampled every half second (`gpu-mib.txt`, a lower bound on the device peak). For CPU steps, each tool's
  `/usr/bin/time -v` peak RSS.
- **CPU steps** (`icheck`, `vstage`, `vcap`, `oreplay`, `lean`): each tool's `/usr/bin/time -v` wall, median and [min, max]
  over the repeats. Lean's verify also splits into `inputs`, `setup` and the session's own checks where it prints them.
  85 runs the V* stages (`vstage`'s per-statement stages, `lean`'s sessions) `JOBS` at a time (`xargs -P`, 4 here), so
  each one's wall is under its neighbours' load, and the step's whole wall (`lean step`) is what an auditor waits.
- **Sizes**: the V* proofs' bytes (`proof_bytes`, the outer sessions' proofs: what the auditor receives), the inner
  proof's (the prover's witness, never sent), the transcript bytes each way, and the staged files (`sizes.txt`).
- **Setup** (`build`): the step's wall and peak once, cold or cached as the job found it (the binaries' hashes and
  whether 85 rebuilt them are in its records). Never repeated: a second build is a cache hit.

What the numbers leave out: the firewall's release side (above); network latency (every verifier is on loopback on the
same node, on the job's other cores); queue and lease waits (the lease's own lines are kept, not counted); and contention:
node 1's queue jobs share cores, and a session's `prover_cores_busy_before` (the prover's cores' load just before it
started) says how loaded they were.
