# Candidate B: direct constraints (`verity-direct`)

The transition unit (one 16-product `mma.sync` update with an arbitrary finite accumulator) as an AIR whose
constraints are the WP2 gadget contracts (`backends/numerical/python/verity_numerical/checker/`), proved today with
Plonky3 `uni-stark` over Goldilocks. The design is `PROTOCOL.md`; this file is how to build, test and run what exists.

Labels on every number from this crate: `proof_class = NON_ZK_PROOF_DIAGNOSTIC`, `authentication = excluded`,
`unit = transition`, `domain = finite`. The PCS is Plonky3 FRI + Keccak Merkle, **not** the Ligero-style interactive
PCS of `PROTOCOL.md`: it is the fastest complete non-ZK proof of the *right* constraints, not Candidate B's end state.

## Layout

- `Cargo.toml`: standalone crate, Plonky3 pinned `=0.4.3-succinct`.
- `src/params.rs`: `REAL` (SIG 8 / EXP 8 / ACC_SIG 24 / WIDTH 25 / FLOOR −132 / groups 8+8), `MAX_DEG = 5`.
- `src/program.rs`: the 12 WP2 gadget contracts written once against a `Ctx` trait (hint, eq, boolean, rng, lookup, mat).
- `src/ctx.rs`: `IntCtx` (i128 witness generator + integer checker) and `AirCtx` (symbolic constraint emitter: range
  checks as 2-bit limbs, fixed tables as one-hot selectors, auto-materialisation above degree 5).
- `src/air.rs`: `UnitAir`, `Layout`, trace generation (one row per unit), per-constraint `Evaluator`.
- `src/stark.rs`: Goldilocks / `Goldilocks^2` challenges, Keccak256, FRI `log_blowup=2, num_queries=100, pow_bits=16`.
- `src/challenger.rs`: fork of `p3_challenger::SerializingChallenger64` with the mask fix below.
- `src/main.rs`: `bench` (writes `result.json` per `verity_numerical.bench.contract`), `crosscheck`, `layout`.
- `tests/negative.rs`: the `test_negative.py` patterns as Rust tests, plus prove/verify of mutated traces.
- `tools/crosscheck.py`, `fixtures/crosscheck_units.json`: 1,200 WP2-generated units for the cross-check.

## What is constrained

One row per unit; rows are independent (no transition constraints yet), so `B` units are `B` rows padded to a power
of two (min 2). Width 2,794 columns: 272 words/hints/named materialisations, 1,828 range-limb columns (2-bit limbs —
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

## Recorded runs (vy-cpu, AMD EPYC 9655P, 16 threads, rustc 1.98.1, release)

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

Not yet built: the VU chain (`c_{i+1} = y_i` between rows, step-0 selector, boundary binding), the Ligero PCS,
HVZK (no randomised columns, no blinded openings), authentication.
