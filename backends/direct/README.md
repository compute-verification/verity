# Candidate B: direct constraints (`verity-direct`)

The transition unit (one 16-product `mma.sync` update with an arbitrary finite accumulator) as an AIR whose
constraints are the WP2 gadget contracts (`backends/numerical/python/verity_numerical/checker/`), and the **full
K=1536 verification unit** as 96 chained rows plus the epilogue (`ChainAir`, PROTOCOL.md 12), proved with Plonky3
`uni-stark` over Goldilocks or BabyBear; range checks and fixed tables either as 2-bit limbs / one-hot selectors
(PROTOCOL.md 12) or as LogUp lookups against preprocessed tables (`--lookup`, PROTOCOL.md 14: 3.4x fewer committed
bytes per VU, 2.9x faster at 4 096 VUs).  The BabyBear port was checked against the red team's six embedding rules
and 83-case battery (PROTOCOL.md 13; one soundness fix, `LT_DIRECT` two-range form in `leading_bit`). The design is `PROTOCOL.md`; this file is how to build, test and run what exists.

Labels on every number from this crate: `proof_class = NON_ZK_PROOF_DIAGNOSTIC`, `authentication = excluded`,
`unit = transition` (`bench`) or `unit = vu`, `scope = vu` (`vu-bench`), `domain = finite`. The PCS is Plonky3 FRI + Keccak Merkle, **not** the Ligero-style interactive
PCS of `PROTOCOL.md`: it is the fastest complete non-ZK proof of the *right* constraints, not Candidate B's end state.

## Layout

- `Cargo.toml`: standalone crate, Plonky3 pinned `=0.4.3-succinct`.
- `src/params.rs`: `REAL` (SIG 8 / EXP 8 / ACC_SIG 24 / WIDTH 25 / FLOOR −132 / groups 8+8), `MAX_DEG = 5`.
- `src/program.rs`: the 12 WP2 gadget contracts written once against a `Ctx` trait (hint, eq, boolean, rng, lookup, mat).
- `src/ctx.rs`: `IntCtx` (i128 witness generator + integer checker) and `AirCtx` (symbolic constraint emitter: range
  checks as 2-bit limbs, fixed tables as one-hot selectors, auto-materialisation above degree 5; in
  `Realisation::LogUp` both become recorded queries instead).
- `src/air.rs`: `UnitAir` (one independent row per unit) and `ChainAir` (one row per unit, accumulator carried by a
  transition constraint, step counter + `first`/`last` selectors, epilogue on every row, claim column bound on the
  last row); `Layout`, trace builders (`chain_trace` is parallel over VUs), per-constraint `Evaluator`.
- `src/ctx.rs`: `FieldKind::{Goldilocks, BabyBear}`; over BabyBear every 32-bit word is two 16-bit limbs (`Word32`).
- `src/stark.rs`: `Backend` trait -- `GoldilocksBackend` (Goldilocks^2 challenges, the forked challenger) and
  `BabyBearBackend` (BabyBear^4 challenges, upstream `SerializingChallenger32`); Keccak256; FRI `log_blowup=2,
  num_queries=100, pow_bits=16` by default.
- `src/challenger.rs`: fork of `p3_challenger::SerializingChallenger64` with the mask fix below.
- `src/main.rs`: `bench`, `crosscheck`, `layout` (the unit); `vu-bench`, `vu-neg`, `chain-layout` (the VU chain,
  `--field goldilocks|babybear`, reading `bench-instances/v1` arrays from `--root`).
  `check-rows` evaluates every AIR constraint on supplied adversarial rows (PROTOCOL.md 13; driven by
  `tools/battery_adapter.py`, the red team's 83-case battery).  `--lookup` on `chain-layout`, `vu-bench`,
  `check-rows` selects the LogUp realisation (PROTOCOL.md 14; `--ext 5` for BabyBear^5 challenges).
- `src/lookup.rs`, `src/lookup_stark.rs`: the LogUp range/table lookups (tables, queries, helper columns, the
  auxiliary round) and the three-round fork of `p3_uni_stark::prove/verify` they need; `air.rs::LookupChainAir`;
  `prove_lookup_bound` / `verify_lookup_bound_bytes` (statement-bound, PROTOCOL.md 14.7).
- `src/statement.rs`: the statement, the header, the barycentric public-column check, the LogUp header and table digest.
- `src/rtt.rs`: the latency hooks (`RttChallenger`, `--rtt-ms`).
- `vendor/p3-fri`: patched `p3-fri` (openings absorbed before α; PROTOCOL.md 14.7).
- `run_vu.sh`: the pod script (build, `cargo test`, `chain-layout`, `vu-neg`, battery, `vu-bench`);
  `run_lookup.sh`: the same for the LogUp realisation (`tests/lookup.rs` incl. the 2^17-row round trip, battery
  `--lookup`, `vu-bench --lookup`).
- `tests/negative.rs`: the `test_negative.py` patterns as Rust tests, plus prove/verify of mutated traces.
- `tests/chain.rs`: two real VUs (from the committed negative arrays, with their correct words) prove and verify over
  both fields; a wrong claimed word, a broken link, a non-zero `c_0`, a skipped step are rejected by `verify`.
- `results/`: the `result.json` and `vu_neg.json` records of the runs in the table below.
- `tools/crosscheck.py`, `fixtures/crosscheck_units.json`: 1,200 WP2-generated units for the cross-check.

## What is constrained

`UnitAir`: one row per unit; rows are independent, so `B` units are `B` rows padded to a power of two (min 2).
`ChainAir` adds to the same row program the chain columns (PROTOCOL.md 12.1) for a width of 2,844 (Goldilocks) /
2,900 (BabyBear). The unit's width 2,794 columns: 272 words/hints/named materialisations, 1,828 range-limb columns (2-bit limbs —
the fallback for a lookup argument `uni-stark` 0.4.3 lacks), 610 one-hot selectors for the four fixed tables
(POW, ALIGN, LEAD, NORM: boolean + sum-to-one + match + range-checked slack), 66 exclusion inverses, 18 slack, 13
materialised products. 3,090 constraints, max degree 5 (`log_quotient_degree = 2`), no multiplicity columns.

Against the census (`security/census.py`: 1,064 committed slots + 306 helpers = 1,370 cells per unit) the trace is
2.04× wider; the excess is entirely the range-check fallback (1,828 limb columns for 306 lookups).

## Build, test, cross-check

Never on a laptop; on the `vy-cpu` pod (`~/.research/machines.toml`):

~~~sh
CARGO_TARGET_DIR=/workspace/target-direct cargo test --release        # 23 tests
cargo run --release -- layout                                          # column/constraint census
python3 tools/crosscheck.py                                            # WP2 gen_step vs Rust hints
~~~

Cross-check on record: 1,200 random finite units — Rust hint columns byte-identical to WP2 `gen_step`, WP2
`check_step` accepts the Rust hints on 1,200/1,200, all 1,200 satisfy the AIR with WP2's `y`. Every
`test_negative.py` pattern is rejected, and mutated traces (`g0.max.E+1`, `y` sign flip, `align_state.q−1/r=1`,
`sum.sgn` flip, `norm.M+1`) prove but fail `p3_uni_stark::verify`.

## Upstream soundness finding

`p3_challenger::SerializingChallenger64::sample` computes `(1 << log_size) - 1` with `log_size = 64` for Goldilocks;
in release builds the shift wraps and the mask is `0`, so **every sampled challenge is zero** (alpha, zeta, FRI betas).
Only the last constraint survives the folding and degree ≥ 3 constraints are unenforced: mutated traces verified.
`src/challenger.rs` forks the challenger with the mask fixed; the negatives fail only after the fix. Anyone using
Plonky3 0.4.3-succinct with a 64-bit field should check this. Not yet reported upstream.

## The transition unit alone: recorded runs (vy-cpu, AMD EPYC 9655P, 16 threads, rustc 1.98.1, release)

~~~sh
uv run --package research research run --on vy-cpu --project verity --source . --stage direct.unit -- \
    cargo run --release -- bench --units 4096 --out result.json
~~~

| B | run id | rows | prove_s | verify_s | proof bytes | rounds | peak RSS | committed elements | census |
|---|---|---|---|---|---|---|---|---|---|
| 1 | r20260922-015957-ceff | 2 | 0.0075 | 0.128 | 2,368,144 | 6 | 12.0 MB | 5,604 | 1,370 |
| 64 | r20260922-020008-f960 | 64 | 0.037 | 0.130 | 2,492,304 | 11 | 14.5 MB | 179,328 | 87,680 |
| 4096 | r20260922-020018-8d43 | 4096 | 1.47 | 0.132 | 2,746,896 | 17 | 462 MB | 11,444,224 (+32,768 quotient) | 5,611,520 |

Witness generation is not in `prove_s`: 4.7 s at B=4096, single-threaded. Rounds are FRI commit phases + trace/quotient
commits as run (the interactive Ligero design is 3). Soundness of this FRI configuration: proven ≈ 83.8 bits (unique
decoding, −log2((1+ρ)/2) per query + PoW), conjectured 216; field terms (≈ 2^-128 · degree) omitted. Proof size is
100 queries × 2,794-wide openings, hence flat in B.

## Latency axis (lane latency, 2026-09-22)

`rounds.sequential_depth` is *measured*: `src/rtt.rs` wraps the Plonky3 challenger and counts every `observe -> sample`
boundary of the prover's transcript as one verifier message of the interactive reading (as run, Plonky3 is Fiat-Shamir,
depth 1); `--rtt-ms X` (or `VERITY_RTT_MS`, `RTT_MS=` in `run_vu.sh`) sleeps `X` ms at each boundary, and the envelope
carries `t.e2e.rtt_Xms` measured plus the other RTTs predicted (`t.total` is compute with the sleep subtracted;
`direct.depth_formula_legacy` is the earlier `log_n + 5`). Measured: 2^13 rows (B=64) depth **16** (legacy 18), 2^19 rows
(B=4096) depth **22** (legacy 24); B=64 e2e at 50 ms RTT 3.98 s = 3.04 compute + 0.14 verify + 16 × 50 ms (model 3.98 s).
See `backends/numerical/reports/latency.md`.

## Statement binding (lanes soundness-fixes + b-bind, 2026-09-22)

A B proof is a proof *of a statement* (PROTOCOL.md 12.5, 14.7): `Statement { words, operands?, min_rows }` (every
claimed `y16`, optionally `x`/`w` with `--bind-operands`). Both realisations prove and verify through bound entry points
-- `stark::{prove_bound, verify_bound_bytes}` (limbs) and `lookup_stark::{prove_lookup_bound, verify_lookup_bound_bytes}`
(LogUp) -- which absorb a header (domain tag, AIR id, field, widths, extension degree, `B`, `K`, `STEPS`, rows,
`degree_bits`, FRI parameters, every word, operand hashes; for LogUp also the preprocessed/aux widths and the SHA-256
of the verifier's table set) before the first challenge, fix `degree_bits` from the statement, and check the `vu.y16`
column (and the operand columns when bound) at ζ, ζ·g against the verifier's own barycentric evaluation. `vu-bench`
(both modes) asserts inline that the wrong-word, wrong-`B` (and, for `--lookup`, substituted-table) statements are
rejected; `--unbound` runs the pre-fix path for a same-box timing baseline and labels the result
`statement_binding.bound = false`. Pins: `tests/chain.rs::statement_suite`, `tests/lookup_statement.rs` (`--include-ignored`,
2^17 rows). `vendor/p3-fri`: `p3-fri 0.4.3-succinct` with the ζ-openings absorbed before the PCS batching challenge α
(`[patch.crates-io]`; the transcript differs from upstream). Gate: `run_bind_gate.sh [check|test|gate|battery|timing|all|full]`
(`vu-neg --lookup`, both batteries, the 4096-VU bound/unbound timing pairs); records in `results/b-bind/`.

| LogUp, 4096 VUs, 2^19 rows, vy-cpu3 12 threads (`r20260922-094310-af1a`) | bound #1 | unbound #1 | bound #2 | unbound #2 |
|---|---|---|---|---|
| BabyBear^4 prove s | 199.4 | 197.6 | 182.8 | 201.8 |
| BabyBear^5 prove s | 213.3 | 225.4 | 217.5 | 218.1 |

The binding is free to within the box's swing (vy-cpu3 is ~3.2x slower than the vy-cpu2 of PROTOCOL.md 14.5).

## The K=1536 VU chain: recorded runs (vy-cpu2, AMD EPYC 9654, **12 threads**, release)

`bench-instances/v1`, tier `vu-k1536`, ids 0..4096 (operand arrays built on the pod by run r20260922-052410-8160,
manifest sha256 `38fb2c3a...`). Scope `vu`, `NON_ZK_PROOF_DIAGNOSTIC`, authentication excluded, 12 of the pod's 32
vCPUs (`RAYON_NUM_THREADS=12`, the pod is shared).

~~~sh
uv run --frozen research run --on vy-cpu2 --project verity --source . --stage b-chain.vu.goldilocks \
    --env RAYON_NUM_THREADS=12 --require-result -- bash backends/direct/run_vu.sh goldilocks 4096
~~~

Correctness gate (every run repeats it): the 23 unit negatives and the 3 chain tests pass; **all 4,096 output words and
all 393,216 intermediate accumulators equal the recorded ones** (the trace's `pack.y` per row against `vu-k1536.acc.u32`,
the epilogue word against `vu-k1536.y.u16`); **all 52 `vu-k1536-neg`** are rejected by witness generation *and* the
rows it produces violate a field constraint (44 `wrong`: `vu.out` on row 95; 8 `reject`: `a[i].t.excl` /
`acc.t.excl`, the finiteness exclusions), over both fields; the proof of the honest trace verifies. Run ids:
r20260922-054320-2c31 (gate only, Goldilocks; the bench step failed on a missing `/usr/bin/time`),
r20260922-055326-699d (Goldilocks), r20260922-055900-f53a (BabyBear), r20260922-060503-6847 (BabyBear, 166 queries).

| field | run id | width | witness s | prove s | **total s** | **s / VU** | verify s | proof bytes | depth | peak RSS | committed / VU | FRI bits (proven / conj.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Goldilocks | r20260922-055326-699d | 2,844 | 24.17 | 158.48 | **182.65** | **0.0446** | 0.148 | 3,231,120 | 24 | 59.8 GB | 273,024 cells, 2.18 MB | 83.8 / 216 |
| BabyBear | r20260922-055900-f53a | 2,900 | 22.05 | 149.01 | **171.06** | **0.0418** | 0.257 | 2,117,836 | 24 | 30.6 GB | 278,400 cells, 1.11 MB | 83.8 / 216 |
| BabyBear, 166 queries (2^-128 proven) | r20260922-060503-6847 | 2,900 | 21.22 | 146.90 | **168.12** | **0.0410** | 0.425 | 3,453,676 | 24 | 30.6 GB | 1.11 MB | 128.6 / 348 |

Trace: 2^19 rows (393,216 real + zero-VU padding) x width. `overhead.vs_native_peak` (t.total / 4096 over 9.85 ps):
**4.53e9** (Goldilocks), **4.24e9** (BabyBear). Ledger: `backends/reports/ledger/b-chain.jsonl`.

Reading the numbers:

- The chain costs 96 rows per VU, so per-row throughput is what wave 1 measured: 524,288 rows in 158 s at 12 threads
  vs 4,096 rows in 1.47 s at 16 threads -- 0.30 ms/row vs 0.36 ms/row; the transition constraints and the epilogue
  are free at this width.
- Witness generation (integer program + i128 checker per row, parallel over VUs) is 13% of the total; it was 76% of the
  wave-1 total because it was single-threaded then.
- **Goldilocks -> BabyBear on this CPU: 1.07x** on the total (1.06x on the prover), with **0.51x memory** and **0.66x
  proof bytes**. tensor.md's 1.6x is for a hashing-bound prover (GPU, bytes committed halve); this CPU prover halves
  the committed bytes (2.18 -> 1.11 MB/VU) and barely moves, so on 12 EPYC threads it is not hashing-bound: the
  quotient evaluation (3,215 constraints x 2^21 LDE rows) and the DFTs dominate. The field flip is *correct* on the
  real prover (the same gate passes); the speed-up it promises is a GPU statement, not measured here.
- Soundness: the default FRI configuration is 100 queries at rate 1/4 with 16 PoW bits: **83.8 bits proven**
  (unique-decoding bound, -log2((1+rho)/2) per query), 216 conjectured; challenges in Goldilocks^2 (128 bits) or
  BabyBear^4 (124 bits); DEEP/batching terms (~degree / |challenge field|) omitted. 2^-128 proven needs 166 queries
  at the same blowup: the third row measures it -- the prover is unchanged (queries only open more Merkle paths),
  measured: the total is unchanged within noise (168.1 s vs 171.1 s), the proof grows 1.63x (3.45 MB) and the verifier 1.65x (0.425 s).
- The claimed output words are committed trace cells (the `vu.y16` column) bound to the epilogue word on each VU's
  last row; nothing binds them to an external statement (authentication excluded, as in wave 1).

Not yet built: the Ligero PCS of PROTOCOL.md 3-5, a range-check lookup argument (1,868 of 2,844 columns are 2-bit
limbs), HVZK (no randomised columns, no blinded openings), authentication, the GPU prover.
