# Candidate A build lane — wave 1 report (2026-09-22 ~02:10 UTC)

Branch `lane/gkr-unit` (worktree `../verity-gkr-unit`, from main `34b0b68`). Commits, oldest first:

- `a0ebc4c` WP2 checker exported as a layered circuit + Rust GKR/logUp prover-verifier (first cut)
- `30493cb` mask deeper wires per layer input (`V^(k)` holds depth <= k only); fix ext test constant
- `8d67c52` measured-run script, thread override; verified on vy-cpu: honest B=1..4096 accepted, 52 negatives rejected
- `10b4cce` Ligero-style column commitment (RS k=4096 n=16384, SHA-256 Merkle, t=192) for W + multiplicities; residual GKR/logUp claims -> one batched linear functional; PCS mode default, `--clear` keeps the arithmetic diagnostic
- `05da1dd` ligero: pad the functional to rows*k
- `27f1057` dishonest witness whose lookup transcript fails yields a proof without opening (verifier rejects earlier)
- `29a8e56` gkr_export: ruff --fix

## Files

- `backends/gkr/` (Rust, no external crates; hand-rolled Goldilocks, cubic extension, SHA-256):
  `src/field.rs` (Fp, Ext = Fp[X]/(X^3-X-1), eq tables, Lagrange), `src/transcript.rs` (SHA-256 Fiat–Shamir, rejection sampling),
  `src/circuit.rs` (circuit text parser, layers), `src/gkr.rs` (data-parallel sumcheck GKR per layer),
  `src/logup.rs` (logUp fractional-sum GKR per table), `src/ligero.rs` (NTT, RS encode, Merkle, open/verify),
  `src/main.rs` (prove/verify, `run` -> result.json, `negatives`, tests), `run_measure.sh`, `fixtures/` (circuit, 8-unit honest witness, 52 negatives).
- `backends/numerical/python/verity_numerical/gkr_export/` (added, nothing under `checker/` edited): `circuit.py` runs the 12 WP2
  gadgets once on symbolic values (`CircuitCtx`) and emits the layered circuit; `negatives.py` re-derives every `test_negative.py`
  case as a mutated witness; `__main__.py` writes fixtures. Tests: `backends/numerical/tests/gkr_export/test_export.py`.

## What is proved — label `NON_ZK_PROOF_DIAGNOSTIC`, `authentication = excluded`

Shipped commitment layer: option (1). The witness matrix `X` = [468 columns/unit x B units | logUp multiplicities of all 5 tables]
is Reed–Solomon encoded row-wise (k=4096 -> n=16384 over Goldilocks), each of the n columns SHA-256-hashed into a Merkle tree;
root absorbed into the transcript before any challenge. No ZK masking (no random rows, no salted leaves). The `--clear` mode
(`ARITHMETIC_DIAGNOSTIC`, runs r20260922-014657-999b / 014821-560c / 014938-14b3 at commit 8d67c52) is kept only as a diagnostic.

- **Circuit.** Exported from the WP2 gadget contracts (same hints, same range checks): 468 committed columns/unit (34 public words +
  hints + range chunks + looked-up rows + materialised query wires), 826 wires, 358 quadratic products, 353 linear assertions,
  374 lookup queries/unit (R16: 352, ALIGN: 16, POW/LEAD/NORM: 2 each), multiplicative depth 4. Non-power-of-two ranges are two
  one-sided checks. The circuit is data-parallel over B units (units are the low hypercube variables).
- **GKR.** Sumcheck over the layered circuit (output layer = all assertions and all products equal claimed wires, random
  linear combination of the assertion vector, then layer by layer down to the input layer). Degree-3 rounds, 2 claims merged per layer.
- **Lookups.** LogUp / fractional-sum GKR per table (Haböck; Papini–Haböck). Table-side leaves are `m_t/(z - t)` with committed
  multiplicities `m`, witness-side leaves `1/(z - q)`; the two fractional sums are proved equal via a binary tree of sumchecks.
  Hypercube dims at B=1: R16 2^17, ALIGN 2^10, POW 2^10, LEAD/NORM 2^6 (grow with B).
- **Commitment.** All residual evaluation claims on the input layer (per GKR layer claim and per table leaf claim) are linear
  functionals of `X`; they are batched with one random combination into one functional `a` and opened Ligero-style: prover sends
  the row-combination `r^T X` restricted to the k-symbol payload plus the t=192 opened columns with Merkle paths; verifier checks
  proximity (opened columns match the encoded combination) and the functional value.
- Fiat–Shamir with SHA-256 everywhere; `rounds.protocol` counts prover->verifier messages, `rounds.sequential_depth` is the longest
  chain of challenge dependencies.

Soundness (`security/accounting.py`, `account_candidate_a` with these parameters, `LigeroParams(n=16384,k=4096,l=4095,t=192,zk=False)`,
`leaf_commitment_bits=None`, `commit_query_wires=True`): **127.7 bits at B=1, 64, 4096**. Ligero terms: IRS query 2^-130.2,
linear/quadratic query 2^-192, field terms <= 2^-178. Note: my opening does a *linear* test only (no quadratic test is needed since
all products are inside the GKR), so the quadratic term is slack. The 0.3 bits below 2^-128 come from the accounting's logUp
identity term over Goldilocks^3; parameterised, not tuned.

## Cross-checks vs WP2

- Honest units: **5120** random finite units (`random_finite_unit`, WP2 `gen_step`), each accepted by WP2 `check_step` (Python)
  and by the exported circuit's own evaluator (`verify_unit`), in two fixture sets (seed 20260921: 1024 units; seed 4096: 4096 units).
  The Rust verifier accepted the 4096-unit batch end-to-end (run bb50 below) and the 1024-unit file at B=1 and B=64.
  `fixtures/bench-instances/v1` is not on main (checked `git show main:...`), so honest units are WP2 `gen` on random finite inputs.
- `cargo test --release` on vy-cpu: 9/9 green (field, NTT/encode, SHA-256 vectors, honest batches B=1..64, tampered proof, all negatives).
- Python: `pytest backends/numerical/tests/gkr_export` 3/3 green; `pytest backends/numerical -q` 291/291 green; ruff clean on the added files.

## Negatives rejected (52/52, 0 wrongly accepted; `verity-gkr negatives`, PCS mode, commit 27f1057)

align_product.{steal_unshifted, under_quotient, wrapped, wrong_r}; align_state.{flushed_claims_quotient, flushed_claims_quotient_keep_r,
r_plus_one, wrapped_remainder}; bf16.{non_finite_7f80, non_finite_7f81, non_finite_7fc0, non_finite_ff80, nz_cleared, nz_set_on_zero,
sign_flip, wrapped_reconstruction}; fp32.{false_nonzero, false_zero, sign_flip, subnormal_hidden_bit, wrapped_reconstruction};
group_max.{below_floor, excessive, excessive_floor_selector, false_selector, two_selectors}; leading_bit.{minus_one, plus_one,
zero_claims_bit}; normalise.{flip_clamp, flip_clamp_subnormal, left_shift_steals_remainder, m_plus_one, m_plus_one_wide, wrapped};
output.{plus_one, sign_flip}; overflow.{max_finite, neg_inf, zero}; pack.{hidden_only, wrapped}; product.{claim_nonzero, claim_zero};
signed_sum.{claim_nonzero, claim_zero, negative_zero, sign_flip}; state_out.{claim_nonzero, claim_overflow, claim_zero, deny_overflow}.
Each fails either at a logUp final identity (range/table violation) or at the assertion sumcheck.

## Measurements (vy-cpu, 16 threads, `unit=transition`, K=16, commit 27f1057, `NON_ZK_PROOF_DIAGNOSTIC`, `authentication=excluded`)

| B | run id | t.prove_s | t.verify_s | proof_bytes | rounds.protocol | rounds.sequential_depth | mem.peak_rss_bytes | committed_elements | ligero rows |
|---|---|---|---|---|---|---|---|---|---|
| 1 | r20260922-020045-0ac2 | 0.250 | 0.082 | 443,864 | 684 | 172 | 37,040,128 | 67,091 | 17 |
| 64 | r20260922-020115-8b4e | 0.331 | 0.106 | 461,432 | 762 | 172 | 42,831,872 | 96,575 | 24 |
| 4096 | r20260922-020128-bb50 | 6.29 (commit 0.21, lookup 2.39, arith 2.05, open 1.55) | 1.56 | 1,206,296 | 1177 | 254 | 644,718,592 | 1,983,551 | 485 |

`committed_elements` = 468·B witness columns + ~66,623 multiplicity entries (R16 has 65,536 rows) — the table multiplicities dominate
at small B. Per unit: 468 columns vs census 321 committed + query wires (my export materialises 147 extra columns: range chunks and
query wires bound by assertions, as PROTOCOL.md 4.3 `commit_query_wires`). Clear-mode reference (`ARITHMETIC_DIAGNOSTIC`,
commit 8d67c52): B=1 prove 0.132 s / verify 0.010 s / 36,816 B (r20260922-014657-999b); B=64 r20260922-014821-560c;
B=4096 r20260922-014938-14b3. The proof is dominated by the Ligero opening (192 columns x rows x 8 B + paths); the ~440 KB floor at B=1
is the price of committing the 65,536-row R16 multiplicity vector.

## What is missing

- **VU chain (word-chained units):** units are independent here; the chain needs the output accumulator `y` of unit i bound to the
  input `c` of unit i+1 (one extra linear assertion across adjacent units in the data-parallel index — a shifted-copy wire in
  the GKR input layer) and the public step-0/step-K boundary as public inputs. Not built.
- **PCS:** no ZK; no quadratic test (not needed); single-repetition field tests (sigma=1 in Ext); columns are hashed unsalted;
  Merkle paths are not batched (proof bytes could drop ~3x); `t=192` fixed, not derived from the target per batch; NTT is single-threaded.
- **HVZK masking:** none. Needs random rows in the Ligero matrix (k > l + t), masked sumcheck polynomials (Libra-style masking
  polynomials for GKR and for the fractional-sum tree), and salted leaves (HM96 term). The accounting object already parameterises these.
- **Step 0:** no coin commitment (malicious-verifier ZK); Fiat–Shamir only; no explicit interactive transcript object (the transcript
  does record `rounds.*`).
- Threads: prover parallelises over units/tables (16 threads); verifier single-threaded. Peak RSS at B=4096 is 645 MB.

## What would kill Candidate A on this evidence

- Verifier time: 1.56 s at B=4096 is dominated by re-encoding the row-combination (rows x NTT of 16384) and 192 Merkle paths; if the
  target needs a sub-100 ms verifier at millions of units, the Ligero opening must be replaced by a Brakedown/FRI-style PCS.
- Proof size: ~1.2 MB at B=4096 and a 440 KB floor; any target requiring < 100 KB proofs kills this configuration.
- Committed columns per unit are 468, not 321: the census under-counts the range-chunk and query-wire columns the checker actually
  needs; if the cost model is redone with 468 the Candidate A/B comparison shifts.
- Sequential depth 254 at B=4096 (172 at B=1) is far above the `rounds_and_depth_a` estimate (222/258 with c=1 variables per round
  at B=4096 — comparable; but the lookup chains for R16 at 2^21 dominate and grow with log B).

## Overnight sweep (<= 5)

1. `research run` sweep B ∈ {2^k, k=0..14} with `run_measure.sh`, both modes; fit prove/verify/bytes vs B (verify should be ~linear in
   rows = 468·B/4096).
2. Batch the 192 Merkle paths and send the combined row once per functional group; expect ~3x smaller proofs — measure.
3. Sweep `LIGERO.t` and rate (k=4096, n ∈ {8192, 16384, 32768}) against `accounting.ligero_queries_for(-128, ...)`; record bits vs bytes.
4. Verifier profile at B=4096 (`perf` on vy-cpu): split re-encoding vs hashing vs sumcheck bookkeeping; parallelise the NTT.
5. Fuzz: 10k random single-element witness mutations at B=64 through `verity-gkr negatives`-style driver; assert 0 acceptances.
