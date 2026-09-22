# Candidate B, wave 1: the transition unit as direct AIR constraints (Plonky3 uni-stark)

Labels on every number below: `proof_class = NON_ZK_PROOF_DIAGNOSTIC`, `authentication = excluded`,
`unit = transition`, `domain = finite`, `zk_mode = none`. The PCS is Plonky3 FRI + Keccak Merkle, **not**
Candidate B's Ligero-style interactive PCS from `PROTOCOL.md`; it is the fastest route to a complete
non-ZK proof of the *right* constraints. Honest units are WP2 `gen_step` on random finite inputs
(`fixtures/bench-instances/v1` is not on `main`).

## Branch, files

Branch `lane/direct-unit` (worktree `../verity-direct-unit`), 14 commits on top of `main` 34b0b68.

- `backends/direct/Cargo.toml`: standalone crate `verity-direct`, Plonky3 pinned `=0.4.3-succinct`.
- `src/params.rs`: `REAL` params (SIG 8 / EXP 8 / ACC_SIG 24 / WIDTH 25 / FLOOR −132 / groups 8+8), `MAX_DEG = 5`.
- `src/program.rs`: the 12 WP2 gadget contracts written once against a `Ctx` trait (hint, eq, boolean, rng, lookup, mat).
- `src/ctx.rs`: `IntCtx` (i128 witness generator + integer checker with field semantics for exclusion inverses) and
  `AirCtx` (symbolic constraint emitter; range checks as 2-bit limbs, fixed tables as one-hot selectors,
  auto-materialisation above degree 5).
- `src/air.rs`: `UnitAir`, `Layout`, trace generation (one row per unit), per-constraint `Evaluator`.
- `src/stark.rs`: Goldilocks / `Goldilocks^2` challenge field, Keccak256, FRI `log_blowup=2, num_queries=100, pow_bits=16`.
- `src/challenger.rs`: fork of `p3_challenger::SerializingChallenger64` with a mask fix (see "finding").
- `src/main.rs`: `bench` (writes `result.json` per `contract.py`), `crosscheck`, `layout`.
- `tests/negative.rs`: 23 tests mirroring `test_negative.py` + prove/verify of mutated traces.
- `tools/crosscheck.py`, `fixtures/crosscheck_units.json`: 1,200 WP2-generated units.

## What is constrained

One row per unit (row-local AIR; no transition constraints, so rows are independent and `B` units = `B` rows
padded to a power of two, min 2).

| item | count |
|---|---|
| trace width | 2,794 columns |
| of which words + hints + named materialisations | 272 |
| range-limb columns (2-bit limbs, fallback for the missing lookup argument) | 1,828 |
| fixed-table one-hot selectors (POW, ALIGN, LEAD, NORM) | 610 |
| exclusion inverses | 66 |
| saturating-row slack | 18 |
| materialised products | 13 |
| constraints | 3,090 |
| max constraint degree | 5 (limb constraint `x(x−1)(x−2)(x−3)` is degree 4); `log_quotient_degree = 2` |
| lookup argument | none (uni-stark 0.4.3 has none); tables are one-hot selectors with boolean + sum-to-one + match + range-checked slack |
| multiplicity columns | none (no LogUp) |

Census comparison per unit: 1,064 committed slots + 306 helpers = 1,370 cells; actual 2,794 committed cells
(2.04×). The overhead is entirely the range-check fallback: 1,828 limb columns for what the census counts as
306 lookups. Committed cells: B=1 → 5,604 (2 rows), B=64 → 179,328, B=4096 → 11,444,224 vs census 1,370 / 87,680 / 5,611,520.

## Cross-check vs WP2

1,200 random finite units (WP2 `gen_step`, `y = tc_dot`): Rust hint columns are byte-identical to WP2's
`gen_step` hints, WP2 `check_step` accepts the Rust hints on all 1,200, and all 1,200 satisfy the AIR with WP2's `y`.

## Negatives rejected (all 23 tests green on the pod, `cargo test --release`)

`bf16_decode_wrapped_reconstruction`, `bf16_decode_sign_flip`, `bf16_decode_false_nonzero_flag_both_ways`,
`bf16_decode_non_finite_operands`, `fp32_decode_wrapped_reconstruction_and_flags`, `fp32_decode_infinite_accumulator`,
`product_zero_flag`, `group_max_excessive_exponent`, `group_max_false_selector`, `align_state_wrong_remainder`,
`align_state_flushed_state_quotient`, `align_product_wrong_remainder_and_wrapped_quotient`,
`align_product_unshifted_no_remainder`, `signed_sum_sign_flip_and_zero_flag`, `leading_bit_off_by_one`,
`normalise_clamp_flag_and_quotient`, `state_out_flags`, `pack_wrapped_reconstruction_and_output_sign_flip`,
`overflow_output_is_the_infinity`, `every_hint_is_range_checked_against_the_contract`, plus
`mutated_traces_do_not_verify`: mutated traces (`g0.max.E+1`, `y` sign flip, `align_state.q−1/r=1`,
`sum.sgn` flip on a cancelling sum, `norm.M+1`) prove but the proof is rejected by `p3_uni_stark::verify`.

## Finding (soundness, upstream)

`p3_challenger::SerializingChallenger64::sample` computes `(1 << log_size) - 1` with `log_size = 64` for
Goldilocks; in release builds the shift wraps and the mask is `0`, so **every sampled challenge is zero**
(alpha, zeta, FRI betas). Only the last constraint survives the folding and degree ≥ 3 constraints were
unenforced; mutated traces verified. `src/challenger.rs` forks the challenger with the mask fixed; all
negatives fail after the fix. Anyone using Plonky3 0.4.3-succinct with a 64-bit field should check this.

## Measurements (pod `vy-cpu`, AMD EPYC 9655P, 16 threads, rustc 1.98.1, release)

| B | run id | rows | prove_s | verify_s | proof bytes | rounds (protocol = sequential) | peak RSS | committed elements | census_b |
|---|---|---|---|---|---|---|---|---|---|
| 1 | r20260922-015957-ceff | 2 | 0.0075 | 0.128 | 2,368,144 | 6 | 12.0 MB | 5,604 | 1,370 |
| 64 | r20260922-020008-f960 | 64 | 0.037 | 0.130 | 2,492,304 | 11 | 14.5 MB | 179,328 | 87,680 |
| 4096 | r20260922-020018-8d43 | 4096 | 1.47 | 0.132 | 2,746,896 | 17 | 462 MB | 11,444,224 (+32,768 quotient) | 5,611,520 |

Witness generation (not in prove_s): 4.7 s at B=4096, single-threaded i128 `IntCtx`.
Rounds = FRI commit phases + trace/quotient commits as run; the interactive Ligero design would be 3.
FRI: `log_blowup=2, num_queries=100, pow_bits=16`, challenge field `Goldilocks^2`. Soundness: proven bound
(unique decoding, −log2((1+ρ)/2) per query + PoW) ≈ 83.8 bits; conjectured (list decoding up to capacity)
216 bits. Field terms (DEEP/quotient batching over `F_p^2`, ≈ 2^-128 · degree) omitted. Target −80 bits.
Proof size is dominated by 100 queries × 2,794-wide openings, hence flat in B.

## Missing

- **VU chain**: no transition constraints; a word-chained trace needs `c_{i+1} = y_i` between rows (one
  constraint on `next`), a first-row selector for step 0, and either public inputs or a hash-chain for the
  boundary words.
- **Ligero PCS swap**: `stark.rs` is behind `prove_trace/verify_proof`; the AIR itself is PCS-agnostic. Ligero
  needs a quotient-free formulation (linear/quadratic tests per PROTOCOL.md) or its own quotient commitment.
- **HVZK**: no randomised trace columns, no blinding of FRI openings; uni-stark 0.4.3 has no ZK mode.
- **Step 0**: the first unit's `c = 0` case is covered by finite-domain tests but not pinned as a boundary constraint.
- **Authentication**: public words are committed trace cells, not bound to any statement.

## What would kill Candidate B on this evidence

- The 2× census overhead is structural without a lookup argument: 1,828 of 2,794 columns are limbs. If the
  Ligero design cannot host LogUp-style lookups cheaply, the census's 1,370 is unreachable.
- Proof size 2.4–2.7 MB at 100 queries is 3 orders above the target for a transition unit; batching across
  many rows amortises prover time (0.36 ms/unit at 4096) but not proof size.
- Witness generation is 3× slower than proving; a real VU chain needs a vectorised generator.

## Overnight sweep (≤5)

1. LogUp range checks in the same AIR (needs a permutation/lookup-capable STARK or hand-rolled running-sum
   columns): measure width vs 2,794 and prove_s at B=4096.
2. Add the chain constraint `next.c == cur.y` and step-0 selector; re-run negatives with a cross-row mutation.
3. Sweep FRI `(log_blowup, num_queries) ∈ {(1,160),(2,100),(3,70),(4,50)}` for proof bytes vs prove_s at fixed ≥80 proven bits.
4. Parallelise `IntCtx` witness generation over rows (rayon) and record `t.witness` separately at B=65536.
5. Report the challenger bug upstream and test the pinned rev's `DuplexChallenger` path as an alternative.
