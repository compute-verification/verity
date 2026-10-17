# proof_throughput: the proof system's throughput on GLM-5.3's dot products

How many verification units (VU) per second C-Flock proves and the Lean verifier accepts, as a function of the batch size B,
for the dot products GLM-5.3 serves. The report is the throughput frontier over B: one row per B with VU/s, the overhead
against the H200's native peak, the dominant prover phase, peak memory, proof bytes and the Lean verifier's seconds.

~~~text
verity bench proof_throughput workloads                       the VU families and their B caps
verity bench proof_throughput plan  [--workload W] [--store]  each point's identity (and its passing prover run)
verity bench proof_throughput run   [--workload W] [--lo B] [--hi B] [--store] [--dry-run]   one recorded prover run per point
verity bench proof_throughput verify --prove-run RUN          the Lean verifier point on that run's bundle (vy-nebius-cpu-3)
verity bench proof_throughput table RESULT... | --baseline | --store --workload W [--commit C]   the frontier, from results
verity bench proof_throughput baseline [--workload W] [--lock RESULT...]   the pinned hill-climb baseline, or pin one sweep
verity bench proof_throughput compare BASELINE CANDIDATE [--allow FIELD]   refuses results whose identities differ
verity bench proof_throughput zk                              configuration 2: what is measured, stood in for, or stubbed
~~~

## What a point is

A point is one statement of B coordinates of one workload, the first B instances of its frozen set:

1. **Prover run** (`job.sh`, a `research run --queue --on vy-nebius-1 --gpu-host` job, tool `proof_throughput_prover`): builds
   `flock-circuit` with CUDA for sm_120, stages the statement and checks its instances against the manifest outside any GPU
   lease, then proves `WARM` warm and `RUNS` timed sessions on one GPU taken by `gpu-lease` (the kernel-engineering skill's
   sharing rule), with live coins from a loopback `flock-circuit serve` (`FC_COINS=os`). ZK is off. The statement gives every
   coordinate its own rows (`FLOCK_OWN_ROWS=1`), x included: the verifiers refuse a plain statement whose public file shares
   rows (`ProvedScope.sharedRowsOk` covers that only for a registered ZK session), so x is committed once per coordinate
   rather than once per Call, and that cost is in the numbers. It puts the circuit,
   the public file and every session in the evidence store as one bundle (`out/bundle.json`) and writes the contract
   result with the live verdicts only.
2. **Verifier run** (`verify.sh`, a `research run --on vy-nebius-cpu-3` or `-cpu-2` job, tool `proof_throughput_verifier`):
   builds the compiled Lean verifier (`flock-verify`, `verity/core/service/flock/`), runs `flock-verify verify --coins os`
   on every session of the bundle one at a time under `/usr/bin/time`, with its own copy of the `verity/partition/v1` object
   the circuit binds (`--partition`, derived from the workload's program and checked against the circuit's META, not taken
   from the bundle), and writes the point's result of record: the
   prover's measurements and the Lean verifier's. It passes only when the live verifier accepted every session and the
   Lean verifier accepted every timed one.

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
- `verifier.seconds`: the Lean verifier's median wall seconds per timed session; `verify.cpu_s`, `verify.peak_rss_bytes`.
- `fixed.setup_seconds`: staging the statement, once per B. `rounds.protocol`: the live session's rounds.
- The identity (`contract.IDENTITY_FIELDS`): proof class `NON_ZK_PROOF`, profile, K, B, the probed hardware, the security
  target (Fast100, 2 repetitions, 2^-100), plus the workload, its VU, the configuration and the instance set's reference.

`compare` and `table` refuse results that don't conform, didn't pass, or whose identities differ (`table` lets B alone
vary). Tables are rendered on demand from results, never committed. The hill-climb baseline is `baseline.json`: per
workload, the commit one full sweep ran at and each B's verifier run id, no numbers (`baseline --lock` writes it from
the sweep's results, `table --baseline` renders it, and a candidate is `compare`d against the pinned run at its B).

### Native peak

R_native is the H200's dense peak for the workload's number format. The census has no H200 line, and the H200 has the
H100 SXM5's tensor cores and clocks (it differs in memory), so the profiles' anchors are the H100 SXM5's dense peaks:
`bf16-hopper-mma-draft/2026-09-22` (989.4 TFLOPS bf16) and `fp8-hopper-wgmma-draft/2026-09-22` (1978.9 TFLOPS e4m3).

## Workloads

`verity_proof_throughput/workloads.py`, one VU family each, at GLM-5.3's served K:

| workload | VU | served at | cap B |
|---|---|---|---|
| `bf16-k6144` | `GemmCoordinate_v2{K=6144,DOT=HopperBF16WgmmaDot16_v1}` | lm_head, indexer `wk_weights_proj`, MoE router | 1024 |
| `fp8-coord-k6144`, `-k2048`, `-k256` | `ScaledMmFp8BlockCoordinate_v2{K,G=128}` | the block-scaled FP8 GEMMs | declared |

The bf16 sites are those of `integrations/vllm/verity_vllm/program/registry/bf16_gemm_check.py`. The cap is the largest
power of two of coordinates one statement holds at the hill-climb's 2^35 witness bits (each coordinate one 2^25-bit block
at K = 6144). The FP8 coordinates are declared with their profile; the point job stages class-statement workloads only, and
staging them through `benchmarks/served_units/stage.py` is the next workload's work.

## Instances

`proof-throughput/v1`, one tier per workload, pinned by `instances/<workload>.json` (`verity_proof_throughput/instances.py`):
the lanes C-Flock's class sweep draws for the workload's one coordinate class, from one synthetic `Gemm_v2` Call of `n_max`
coordinates at seed 0. Lane i is coordinate i, so B proves ids [0, B) and every B proves a prefix of the same set. The
manifest pins every lane's digest and every ladder prefix's; the prover run recomputes its prefix before proving, and the
result names the manifest by its SHA-256. The words are uniform 16-bit encodings: at K = 6144 every coordinate meets a NaN,
so every output is the same NaN word (`output_words` in the manifest). A Boolean circuit's proving cost doesn't depend on
the values it carries. Regenerate with `instances --workload W --write`.

## Configuration 2: recursive zero knowledge

Not measured yet: `zk` prints its fields. Workers would prove the inner statement natively with ZK off and the outer
proof's arithmetic unmasked; a ZK server (compiled Lean, CPU) would draw the masks and salts, check the outer relations and
release; the auditor runs the Lean verifier. None of that split exists. Its stand-ins, labelled wherever used, are the
firewall (`verity/core/service/firewall`) and `flock-circuit prove --zk` (the masked prover, CPU only). Stubbed: the ZK
server's CPU seconds per session and per VU, the released proof's bytes and the ZK servers needed per GPU. The worker's
throughput is configuration 1's prover at best B (an upper bound), and the bytes from worker to server stand in as
configuration 1's transcript.

## Placement

The prover runs on vy-nebius-1's RTX PRO 6000s (sm_120) as a GPU host job: at most 32 CPUs, the GPU lent by Kueue's pool
through `gpu-lease` around the sessions only. The Lean verifier and other CPU work run on vy-nebius-cpu-3 or -cpu-2, never
vy-nebius-cpu-1.
