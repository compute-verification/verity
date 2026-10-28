# proof_throughput: the proof system's throughput on GLM-5.3's dot products

How many verification units (VU) per second C-Flock proves and the Lean verifier accepts, as a function of the batch size B,
for the dot products GLM-5.3 serves. The report is the throughput frontier over B: one row per B with VU/s, the overhead
against the H200's native peak, the dominant prover phase, peak memory, proof bytes and the Lean verifier's seconds.

~~~text
verity bench proof_throughput workloads                       the VU families and their B caps
verity bench proof_throughput plan  [--workload W] [--store]  each point's identity (and its passing prover run)
verity bench proof_throughput run   [--workload W] [--lo B] [--hi B] [--store] [--dry-run]   one recorded prover run per point
verity bench proof_throughput verify --prove-run RUN [--node N]   the Lean verifier point on that run's bundle (vy-nebius-cpu-3)
verity bench proof_throughput table RESULT... | --baseline | --store --workload W [--commit C]   the frontier, from results
verity bench proof_throughput baseline [--workload W] [--lock RESULT...]   the pinned hill-climb baseline, or pin one sweep
verity bench proof_throughput compare BASELINE CANDIDATE [--allow FIELD]   refuses results whose identities differ
verity bench proof_throughput zk                              configuration 2: what is measured, stood in for, or stubbed
~~~

## What a point is

A point is B coordinates of one workload per session, sampled afresh from its frozen population of `n_max` instances by the
stratified law the Lean verifier's soundness covers:

1. **Prover run** (`job.sh`, a `research run --queue --on vy-nebius-1 --gpu-host` job, tool `proof_throughput_prover`): builds
   `flock-circuit` with CUDA for sm_120, stages the whole population (the same staged statement for every B) and checks its
   instances against the manifest outside any GPU lease, then proves `WARM` warm and `RUNS` timed sessions on one GPU taken
   by `gpu-lease` (the kernel-engineering skill's sharing rule), with live coins from a loopback `flock-circuit serve`
   (`FC_COINS=os`). The server samples each session's B units at its Register by the verifier's law (`serve --draw-law`,
   `law.json`: core's `one_stage.draw.derive` of `stratified:B` over the population's program and its
   `Q_template_instance` partition), and the session proves those units only. ZK is off. The statement gives every
   coordinate its own rows (`FLOCK_OWN_ROWS=1`), x included: the verifiers refuse a plain statement whose public file shares
   rows (`ProvedScope.sharedRowsOk` covers that only with the verifier's own registered reads), so x is committed once per coordinate
   rather than once per Call, and that cost is in the numbers. It puts the circuit,
   the public file and every session in the evidence store as one bundle (`out/bundle.json`) and writes the contract
   result with the live verdicts only.
2. **Verifier run** (`verify.sh`, a `research run --on vy-nebius-cpu-3` or `-cpu-2` job, tool `proof_throughput_verifier`):
   builds the compiled Lean verifier (`flock-verify`, `verity/core/service/flock/`), runs `flock-verify verify --coins os
   --program P --stratified B --partition F` on every session of the bundle one at a time under `/usr/bin/time`, with its
   own copies of the population's program and of the `verity/partition/v1` object the circuit binds (derived from the
   workload, checked against the circuit's META, not taken from the bundle). The verifier derives the stratified law for B
   itself and refuses a session whose sample was taken by any other. It writes the point's result of record: the prover's
   measurements and the Lean verifier's. It passes only when the live verifier accepted every session and the Lean
   verifier accepted every timed one.

**Counted sessions.** A session counts toward throughput only if the Lean verifier accepts it. A sweep that proves the
whole staged set with no sample passes every check, and the verifier refuses it under D12
(`verity/ml/flock/verifier/PROTOCOL.md` §16.15). Such runs are labelled "checks pass, refused under D12 (no sample)", a result of that kind fails with that label in its detail (`verify.lean_drawless`),
and `table` and `baseline` read none of them.

## The result contract

Each point's `result.json` is the campaign contract's, `benchmarks/numerical/python/verity_numerical/bench/contract.py`,
imported as it is and checked by its `validate()` before it is written:

- VU = one output coordinate, `GemmCoordinate<K>`: K MACs (2K FLOP). `vu.count` = B, `vu.per_second` = B / `t.total`.
- `t.total`: the median timed statement's prover seconds (the witness wait plus both repetitions, as the GEMM hill-climb's
  `statement_seconds`); `t.total_live` adds the waits on the live verifier. The contract's buckets are summed over the two
  repetitions: `t.witness`, `t.encoding_commitment`, `t.arithmetic` (zerocheck, lincheck, ring switch, Ligerito, each also
  kept as `prove.*_s`), `t.serialization`. `prove.host_s` is the repetitions' host time around the device.
- `overhead.vs_native_peak` = R_native / R_proved through `contract.rate_measurements` at the workload's profile.
- `mem.peak_host_bytes`: the prover's peak RSS. `mem.peak_device_bytes`: the leased GPU's memory use sampled every 200 ms.
- `proof_bytes`, `transcript.bytes` (`net.bytes_out` from the prover, `net.bytes_in` from the verifier).
- `verifier.seconds`: the Lean verifier's median wall seconds per timed session, one session per call; `verify.cpu_s`,
  `verify.peak_rss_bytes`, and two of its parts, `verify.setup_s` (the sampled statement's setup, which each fresh sample pays
  again) and `verify.session_s` (one session's checks). The rest is the proved-scope check (`ProvedScope.check`), which a
  call runs once, before its first session's clock starts.
- `verifier.cpu_seconds_per_vu` = `verify.cpu_s` / B and `verifier.cores_per_gpu` = `verify.cpu_s` / `t.total`: the
  verifier's CPU per VU, and the cores that keep up with one GPU proving back to back (`contract.verifier_rates`,
  recomputed by `validate`).
- `fixed.setup_seconds`: staging the population, once per point. `rounds.protocol`: the live session's rounds.
- The identity (`contract.IDENTITY_FIELDS`): proof class `NON_ZK_PROOF`, profile, K, B, the probed hardware, the security
  target (Fast100, 2 repetitions, 2^-100), plus the workload, its VU, the configuration, the population's reference
  (`population`) and the sample (`draw`: stratified, k = B). `instances` is null: a sample fixes no id range.

`compare` and `table` refuse results that don't conform, didn't pass, or whose identities differ (`table` lets B alone
vary). Tables are rendered on demand from results, never committed. The hill-climb baseline is `baseline.json`: per
workload, the commit one full sweep ran at and each B's verifier run id, no numbers (`baseline --lock` writes it from
the sweep's results, `table --baseline` renders it, and a candidate is `compare`d against the pinned run at its B).

### Native peak

R_native is the H200's dense peak for the workload's number format. The census has no H200 line, and the H200 has the
H100 SXM5's tensor cores and clocks (it differs in memory), so the profiles' anchors are the H100 SXM5's dense peaks:
`bf16-hopper-mma-draft/2026-09-22` (989.4 TFLOPS bf16) and `fp8-hopper-wgmma-draft/2026-09-22` (1978.9 TFLOPS e4m3).

## The verifier's profile

`verify.sh PROFILE=1` (the verifier point's command with `PROFILE=1` appended) runs one timed session under
`perf record --call-graph dwarf` and writes `out/profile.json` and a `proof-throughput/verifier-profile/v1` result
(`profile.py`). It attributes each sample by stage (the outermost frame among the proved-scope check, the session's setup
and its verification), by phase (the innermost frame that names a protocol step) and by kernel (SHA-512, GF(2^128),
allocation and reference counting, bignums, strings). Its wall and CPU seconds tell how many threads the verifier keeps
busy.

## Workloads

`verity_proof_throughput/workloads.py`, one VU family each, at GLM-5.3's served K:

| workload | VU | served at | cap B |
|---|---|---|---|
| `bf16-k6144` | `GemmCoordinate_v2{K=6144,DOT=HopperBF16WgmmaDot16_v1}` | lm_head, indexer `wk_weights_proj`, MoE router | 1024 |
| `fp8-coord-k6144`, `-k2048`, `-k256` | `ScaledMmFp8BlockCoordinate_v2{K,G=128}` | the block-scaled FP8 GEMMs | declared |

The bf16 sites are those of `verity/ml/definitions/registry/bf16_gemm_check.py`. The cap is the largest
power of two of coordinates one statement holds at the hill-climb's 2^35 witness bits (each coordinate one 2^25-bit block
at K = 6144). The FP8 coordinates are declared with their profile; the point job stages class-statement workloads only, and
staging them through `benchmarks/served_units/stage.py` is the next workload's work.

## Instances

`proof-throughput/v2`, one population per workload, pinned by `instances/<workload>.json`
(`verity_proof_throughput/instances.py`): `n_max` independent instances of the workload's VU Definition (core's
`population_program` at v0, whose digest every lane is derived from, `verity_proof_throughput/population.py`), each
instance one unit of `Q_template_instance` v0, so the population is one stratum and the stratified law samples B of its
units uniformly. The lanes are those C-Flock's class sweep
gives that unit class at seed 0. The manifest pins every lane's digest and the population's; the prover run recomputes
them before proving, and the result names the manifest by its SHA-256. The words are uniform 16-bit encodings: at K = 6144
every coordinate meets a NaN, so every output is the same NaN word (`output_words` in the manifest). A Boolean circuit's
proving cost doesn't depend on the values it carries. Regenerate with `instances --workload W --write`.

## Configuration 2: recursive zero knowledge

Workers would prove the inner statement natively with ZK off and the outer proof's arithmetic unmasked; a ZK server
(compiled Lean, CPU) would sample the masks and salts, check the outer relations and release; the auditor runs the Lean
verifier. That split doesn't exist. The ZK point measures today's stand-in,
labelled so in its result (`proof-throughput/zk-standin/v1`): `job.sh ZK=1` proves the same sampled sessions with
`flock-circuit prove --zk --gpu` (M1's masked protocol) under the ZK server (`FC_FIREWALL=1`, PROTOCOL §10): `flock-circuit`'s
checks on its host threads and the Lean (`verity/core/service/flock/Flock/Firewall`, the `flock-firewall` processes the
prover starts on the same node: the session loop, the frames and the finish, which commits level 0 and computes the
opening and the proof file), and `zk-point` writes the result: the prover process's CPU seconds per session, the
ZK server's recheck and opening times, the session's bytes out and its released proof. The ZK server's CPU per session is
that CPU with `FC_FIREWALL=1` less the same point's with `FC_FIREWALL=0`; `/usr/bin/time` counts a `flock-firewall`
process's CPU only once the prover has waited for it. The ZK server rechecks a witness the host assembled (`FC_HOST_WITNESS=1`,
§10.2), so `ZK=1` sets it in both arms. `zk` prints each field and its source.

## Placement

The prover runs on vy-nebius-1's RTX PRO 6000s (sm_120) as a GPU host job: at most 32 CPUs, the GPU lent by Kueue's pool
through `gpu-lease` around the sessions only. The Lean verifier runs in a proofs-pool slot of vy-nebius-cpu-3 or -cpu-2
(`--pool proofs --cwd clone`), or, while those slots are held by builds, as a queued CPU job of its own on vy-nebius-1
(`verify --node vy-nebius-1`: 4 CPUs in its own pinned scope, no GPU). Never vy-nebius-cpu-1. The result's
`software.verifier` names the binary; the run's machine is in its Attempt.
