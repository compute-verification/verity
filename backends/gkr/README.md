# Candidate A: GKR + lookups (`verity-gkr`)

The transition unit (one 16-product `mma.sync` update with an arbitrary finite accumulator) as a layered arithmetic
circuit exported from the WP2 gadget contracts, proved data-parallel over `B` units with sumcheck GKR, LogUp fractional-sum
GKR for the lookups and a Ligero-style hash commitment; and, since 2026-09-22, the **full K = 1536 VU** (96 chained units
+ epilogue, c_0 = 0 and the output words public) as one proof over one commitment (section "Full VU runs"). The design is `PROTOCOL.md`; this file is how to build, test and
run what exists.

Labels on every number from this crate: `proof_class = NON_ZK_PROOF_DIAGNOSTIC` (PCS mode) or `ARITHMETIC_DIAGNOSTIC`
(`--clear`: inputs and hints sent in the clear), `authentication = excluded`, `unit = transition` or `vu`, `domain = finite`.
No ZK masking anywhere yet; Fiat–Shamir with SHA-256 (the interactive object and step 0 are not implemented; the transcript
records `rounds.protocol` and `rounds.sequential_depth`).

**Latency axis (lane latency, 2026-09-22).** `rounds.sequential_depth` is *measured*: `Transcript::round_trip()` counts
the critical-path verifier messages (the longest of the lock-step chains; the logUp `(mu, lambda)` pair is one message),
and `--rtt-ms X` (or `VERITY_GKR_RTT_MS`, `RTT_MS=` in `run_measure_vu.sh`) sleeps `X` ms at each so the envelope carries
`t.e2e.rtt_Xms` measured and the other RTTs predicted (`t.total` is compute with the sleep subtracted; `depth_lookup_chain`,
`depth_checker_chain`, `depth_formula_legacy` are alongside). Full VU B=4096 Goldilocks: depth **355** (legacy counter 381;
lookup chain 352 = R16 tree of 26 levels, checker chain 197), B=64: depth 214 (lookup 211, checker 167). Measured e2e on
vy-cpu2: B=64 at 50 ms RTT 18.44 s = 7.20 compute + 0.53 verify + 214 × 50 ms (model 18.43 s); see
`backends/numerical/reports/latency.md`.

## Layout

- `Cargo.toml`: standalone crate, **no external crates** (hand-rolled Goldilocks, cubic extension `X^3 − X − 1`, SHA-256).
  `--features babybear` swaps in `src/field_babybear.rs` (BabyBear `p = 2^31 − 2^27 + 1`, degree-6 extension
  `X^6 − 22`, same API) and reads `fixtures/babybear/` (section "BabyBear"; `PROTOCOL.md` §13).
- `src/field.rs` (Fp, Ext, eq tables, Lagrange), `src/transcript.rs` (SHA-256 Fiat–Shamir, rejection sampling),
  `src/circuit.rs` (circuit text parser, layers), `src/gkr.rs` (data-parallel sumcheck GKR per layer), `src/logup.rs`
  (LogUp fractional-sum GKR per table), `src/ligero.rs` (NTT, RS encode, Merkle, open/verify), `src/main.rs`
  (`prove`/`verify` over a multi-segment `Instance` with an optional `Chain`, `run`/`run-vu` → `result.json` per
  `verity_numerical.bench.contract`, `negatives`/`negatives-vu`, tests incl. synthetic chained VUs).
- `run_measure.sh` / `run_measure_vu.sh`: the recorded-run drivers. `fixtures/`: the exported circuit, an 8-unit honest
  witness, 52 negatives, `vu_synth/` (a synthetic 2-VU chain + epilogue for the unit tests).
- The circuit exporter lives campaign-side: `backends/numerical/python/verity_numerical/gkr_export/` (`circuit.py` runs the
  12 WP2 gadgets once on symbolic values and emits the layered circuit; `negatives.py` re-derives every `test_negative.py`
  case as a mutated witness; `vu.py` exports the epilogue circuit and the chained VU witnesses of `bench-instances/v1`
  (`units.bin`, `epilogue.bin`, `public.bin` = the output words) and finalises the result envelope;
  `python -m verity_numerical.gkr_export` writes the fixtures). Tests:
  `backends/numerical/tests/gkr_export/`.

## What is proved

- **Circuit** (from the gadget contracts, same hints and range checks): 468 committed columns per unit (34 public words +
  hints + range chunks + looked-up rows + materialised query wires), 826 wires, 358 quadratic products, 353 linear
  assertions, 374 lookup queries per unit (`R16` 352, `ALIGN` 16, `POW`/`LEAD`/`NORM` 2 each), multiplicative depth 4.
  Non-power-of-two ranges are two one-sided checks. Units are the low hypercube variables.
- **GKR.** Output layer = all assertions and all products equal their claimed wires (random linear combination of the
  assertion vector), then layer by layer down to the inputs; degree-3 rounds, 2 claims merged per layer.
- **Lookups.** LogUp / fractional-sum GKR per table (Haböck; Papini–Haböck): table leaves `m_t/(z − t)` with committed
  multiplicities, witness leaves `1/(z − q)`, equality of the two sums via a binary tree of sumchecks.
- **Commitment.** `X = [468 columns/unit × B | multiplicities of the 5 tables]`, Reed–Solomon encoded row-wise
  (k = 4096 → n = 16384 over Goldilocks), SHA-256 Merkle over the n columns, root absorbed before any challenge. Every
  residual input-layer claim is a linear functional of `X`; they are batched into one functional and opened Ligero-style
  (row combination restricted to the k-symbol payload + t = 192 opened columns with paths). Linear test only: all
  products are inside the GKR, so no quadratic test is needed.

Soundness by `security/accounting.account_candidate_a` with `LigeroParams(n=16384, k=4096, l=4095, t=192, zk=False)`,
`commit_query_wires=True`: **127.7 bits at B = 1, 64, 4096** (IRS query term 2^-130.2; the 0.3 bits under 2^-128 are the
LogUp identity term over Goldilocks^3, parameterised, not tuned).

Against the census (`security/census.py`: 321 committed columns per unit + query wires) the export materialises 147 extra
columns (range chunks and query wires bound by assertions, `PROTOCOL.md` §4.3 `commit_query_wires`).

## Build, test, cross-check

Never on a laptop; on the `vy-cpu` pod (`~/.research/machines.toml`):

~~~sh
CARGO_TARGET_DIR=/workspace/target-gkr cargo test --release        # 9 tests: field, NTT/encode, SHA-256 vectors, honest B=1..64, tampered proof, negatives
cargo run --release -- negatives fixtures/                          # 52 mutated witnesses, all rejected
uv run --package verity-numerical python -m verity_numerical.gkr_export   # regenerate circuit + fixtures
~~~

Cross-check on record: 5,120 random finite units (WP2 `gen_step`) each accepted by WP2 `check_step` and by the exported
circuit's evaluator; the Rust verifier accepted the 4,096-unit batch end-to-end. Negatives: 52/52 rejected, 0 wrongly
accepted, each failing at a LogUp final identity (range/table violation) or the assertion sumcheck — the families are
`align_product`, `align_state`, `bf16`, `fp32`, `group_max`, `leading_bit`, `normalise`, `output`, `overflow`, `pack`,
`product`, `signed_sum`, `state_out`.

## Recorded runs (vy-cpu, 16 threads, commit 27f1057, PCS mode)

~~~sh
uv run --package research research run --on vy-cpu --project verity --source . --stage gkr.unit -- \
    bash backends/gkr/run_measure.sh 4096
~~~

| B | run id | prove_s | verify_s | proof bytes | rounds.protocol | rounds.sequential_depth | peak RSS | committed elements |
|---|---|---|---|---|---|---|---|---|
| 1 | r20260922-020045-0ac2 | 0.250 | 0.082 | 443,864 | 684 | 172 | 37 MB | 67,091 |
| 64 | r20260922-020115-8b4e | 0.331 | 0.106 | 461,432 | 762 | 172 | 43 MB | 96,575 |
| 4096 | r20260922-020128-bb50 | 6.29 (commit 0.21, lookup 2.39, arith 2.05, open 1.55) | 1.56 | 1,206,296 | 1177 | 254 | 645 MB | 1,983,551 |

`committed_elements` = 468·B + ~66,623 multiplicity entries (`R16` has 65,536 rows: the table multiplicities dominate at
small B, and the ~440 KB proof floor at B=1 is the price of committing that vector). Clear mode (`ARITHMETIC_DIAGNOSTIC`,
commit 8d67c52): B=1 prove 0.132 s / verify 0.010 s / 36,816 B (r20260922-014657-999b); B=64 r20260922-014821-560c;
B=4096 r20260922-014938-14b3.

## Full VU runs (vy-cpu2, 12 threads, PCS mode, `unit = vu`, K = 1536)

The full `GemmCoordinate<1536>`: 96 chained transition units from the constant c_0 = 0 plus the epilogue
(`Epilogue.check`, RN-even cast to the BF16 output word), for B VUs of `bench-instances/v1 vu-k1536`, as **one** proof
over one Ligero commitment. Design: `PROTOCOL.md` §12 (data-parallel units + epilogue segment, chain linkage and the
public output words as batched linear constraints on the committed vector; no checkpoint interval, every word is
committed). The proof's public endpoints are exactly the VU's: c_0 = 0 (constant) and the 4096 recorded output words
(verifier inputs, absorbed into the transcript before any challenge). Labels: `NON_ZK_PROOF_DIAGNOSTIC`,
`authentication = excluded`, scope `vu`, security **2^-127.7** (`account_candidate_a` plus the chain functional's
Schwartz–Zippel and batching terms, both < 2^-170; 0.3 bits short of 2^-128 as before).

~~~sh
uv run --package research research run --on vy-cpu2 --project verity --source . --stage gkr.vu -- \
    bash backends/gkr/run_measure_vu.sh 4096                 # positives; NEGATIVES=1 for vu-k1536-neg
uv run --package verity-numerical python -m verity_numerical.gkr_export.vu export --tier vu-k1536 --lo 0 --hi 4096 --out DIR
verity-gkr run-vu --dir DIR --vus 4096 --out result.json    # `negatives-vu --dir DIR` for the negative tier
~~~

| B | run id | commit | prove_s | verify_s | proof bytes | rounds.protocol | rounds.sequential_depth | peak RSS | committed elements | s / VU |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | r20260922-053451-03f9 | 3c359a0 | 6.33 (commit 0.30, lookup 2.26, arith 3.18, open 0.56) | 0.62 | 1,570,760 | 1247 | 279 | 725 MB | 2,942,591 | 0.0989 |
| 4096 | r20260922-053757-3b76 | 3c359a0 | 293.4 (commit 8.7, lookup 90.8, arith 159.2, open 32.9) | 32.6 | 69,576,488 | 1923 | 438 | 33.8 GB | 184,128,575 | 0.0716 |
| 4096 | **r20260922-055358-9f6b** (headline) | 2229c4b | 297.6 (commit 8.9, lookup 92.4, arith 161.4, open 33.0) | 33.0 | 69,576,488 | 1923 | 438 | 33.9 GB | 184,128,575 | 0.0727 |

Both B = 4096 runs accepted all 4096 VUs with the recorded output words (`validation.status = passed`,
`overhead.vs_native_peak` = 7.4e9 in the `research/result/v0.1` envelope). The pod was shared with two other lanes
(load ≈ 35–40 on 32 vCPU) during every run; the first B = 4096 run also overlapped a duplicate launch of itself for
its first three minutes, so 9f6b is the recorded number. Committed elements = 468·96·B + 9·B + multiplicities of the
five tables (`R16` is 2^28 leaves at B = 4096, which sets the lookup chain and hence the depth 438 vs 254 for the
transition-unit batch). Per transition unit this is 0.757 ms (297.6 s / 393,216) against 1.54 ms for the
transition-unit batch (6.29 s / 4096 on 16 threads): the per-proof fixed costs (table multiplicities, the 2^16 `R16`
rows, the opening) amortise over 96× the rows; the epilogue segment and the 401,408 linkage/endpoint constraints are
in the noise (`t.witness` 1.4 s).

**Correctness gate** (all on vy-cpu2): positives above; `vu-k1536-neg` **52/52 rejected, 0 wrongly accepted**
(r20260922-054550-62b1: 8 rejected at witness generation — `acc.t`/`bf16.t = 255` out of the finite domain, the
non-finite / saturated negatives — and 44 by the verifier at the assertion sumcheck, the claimed word contradicting
the epilogue's `y16` assertion and the endpoint constraint); the transition-unit path's `fixtures/negatives`
**52/52 rejected** on the same binary (r20260922-055135-ed34). Witness generation cross-checks every unit and the
epilogue against WP2 `check_step` / `Epilogue.check` and the recorded accumulators before writing.

**Checkpoint sweep.** The design has no interval m (it is the m = 1 point: every accumulator word is already a
committed column). `PROTOCOL.md` §12.4 gives the cost model for m ∈ {1, 2, 4, 8, 16, 32, 96}: committed elements per
VU move from 44,937 (m = 1) to 44,747 (m = 96, the fully sequential deep circuit), i.e. by 0.42 %, while the checker
depth grows from 243 to 27,075 and the checker's wire evaluations from 1.95G to 187G. No point of the sweep can
change the picture, so only m = 1 is built and measured; the two extremes are the m = 1 row above (measured) and the
m = 96 column of §12.4 (cost model, not run).

Not yet built: ZK rows / salted leaves / batched Merkle paths in the commitment (~3× bytes slack), HVZK masking, the
interactive object and step 0, a parallel NTT, `R16` sharding (depth 438 → 409), authentication.

## BabyBear (vy-cpu2, 12 threads, PCS mode, `--features babybear`; lane a-babybear, 2026-09-22)

The same prover over BabyBear `p = 2^31 − 2^27 + 1` with the degree-6 extension `F_p[X]/(X^6 − 22)` (the polynomial
of `tensor/reference.py` and the a-gpu lane), implementing the six embedding rules of
`backends/numerical/reports/babybear_rules.md` in the exporter's `--field babybear` mode (`gkr_export/babybear.py`:
words as 16-bit limb pairs, direct bound ranges on linear wires instead of `lt_pow` products, LEAD `span` / NORM `mbnd`
columns, `NORM_VBND` dropped; per unit **444 columns / 322 products / 322 assertions / 382 lookups** vs Goldilocks
468 / 358 / 353 / 374). `PROTOCOL.md` §13 has the field, the rule-by-rule circuit mapping, the per-limb chain linkage,
the three byte-level protocol changes and the soundness table (degree 6: `2^-130.2` by the accountant at the
Goldilocks Ligero parameters k = 4096, n = 16384, t = 192; label **2^-128**).

~~~sh
cargo test --release --features babybear                                   # 12 tests: field vectors from tensor/reference.py, honest/negatives/synth VUs
uv run --package verity-numerical python -m verity_numerical.gkr_export fixtures/babybear --units 8 --field babybear
uv run --package verity-numerical python backends/gkr/tools/battery_adapter.py      # the 83-case battery through prove+verify
FIELD=babybear GATE=1      bash backends/gkr/run_measure_vu.sh             # cargo test, 52 TU negatives, battery
FIELD=babybear             bash backends/gkr/run_measure_vu.sh 4096        # export (--field babybear) + prove + verify
FIELD=babybear NEGATIVES=1 bash backends/gkr/run_measure_vu.sh             # vu-k1536-neg
FIELD=babybear TRANSCRIPTS=1 bash backends/gkr/run_measure_vu.sh           # transcript vectors B = 4, 64 for lane a-gpu
~~~

**Range-table soundness fix (both fields, commit 445cbca).** While porting, the exporter's single-chunk range check
`y · 2^(16−w) ∈ R16` for `w < 16` turned out to be forgeable in any prime field (`y = (r + k p)/2^(16−w)` passes for
every `r < 2^16`; over Goldilocks `y = p − (p−1)/2^15` passes a bit check). Sub-16-bit ranges now use exact-width
implicit tables `R7/R8/R9/R14` (query count unchanged, 9 logUp instances instead of 5). The Goldilocks circuit was
regenerated and **re-measured on the same box and binary revision** as the control below; the a-chain numbers in
"Full VU runs" (297.6 s, depth 438) predate the fix. The fix also shortens the lookup chain (`R16` lost the
sub-16-bit chunks): depth 438 → **381** at B = 4096.

**Goldilocks vs BabyBear, same commit (54e5da7 / 25d4bcc), same box (shared with b-chain and checker-min, load ≈ 35–40
on 32 vCPU), 12 threads, all recorded words accepted:**

| B | field | run id | prove_s (commit / lookup / arith / open) | verify_s | proof bytes | rounds | depth | peak RSS | committed | s / VU | ratio BB/GL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | Goldilocks^3 | r20260922-064620-cf4c | 5.79 (0.22 / 1.89 / 3.22 / 0.43) | 0.49 | 1,638,920 | 1974 | 234 | 587 MB | 2,959,871 | 0.0905 | — |
| 64 | **BabyBear^6** | r20260922-064308-006f | 6.93 (0.25 / 2.34 / 3.65 / 0.68) | 0.74 | 1,057,496 | 1991 | 234 | 454 MB | 2,812,351 | 0.1083 | prove 1.20×, verify 1.51×, bytes 0.65×, RSS 0.77× |
| 4096 | Goldilocks^3 | r20260922-065844-2b84 | 261.3 (9.1 / 63.8 / 160.1 / 26.1) | 26.1 | 69,693,032 | 3179 | 381 | 33.8 GB | 184,145,855 | 0.0638 | — |
| 4096 | **BabyBear^6** | r20260922-065112-d687 | **319.3** (6.1 / 79.7 / 188.1 / 44.4) | **45.0** | **33,394,904** | 3176 | 381 | **25.7 GB** | 174,704,575 | 0.0780 | **prove 1.22×, verify 1.72×, bytes 0.48×, RSS 0.76×** |

Reading: on this CPU BabyBear^6 is a *slower* prover (1.2×) with half the proof and three quarters of the memory —
inside the ±1.5× band, so no `--breakthrough`. The challenge-field arithmetic dominates the sumchecks (`arith`,
`lookup`): a degree-6 multiply is 36 base products with 31-bit operands against 9 native `u64` products for
Goldilocks^3, and the opening (`open`: the `w`/`q` polynomials and the row combination over the extension) grows
1.7×; the 4-byte base field only helps the commit (6.1 vs 9.1 s) and the bytes. `tensor.md`'s flip to BabyBear was
priced for INT8 tensor cores (limb matmuls), not for scalar CPU arithmetic — this measurement does not contradict it,
it says the CPU reference gains nothing from the field. Per transition unit: 0.812 ms (BabyBear) vs 0.665 ms.

**Correctness gate (vy-cpu2, `--features babybear`):** `cargo test` 12/12 (r20260922-063837-b26c); transition-unit
`fixtures/babybear/negatives` **52/52 rejected**; `vu-k1536-neg` **52/52 rejected** (44 by the verifier at the
assertion sumcheck, 8 at witness generation — the non-finite / saturated negatives whose hints leave the finite
domain; r20260922-064946-e1b6); **the 83-case BabyBear battery 83/83 rejected** through prove + verify
(`tools/battery_adapter.py`, r20260922-063837-b26c: wrap 58, z3 12, zero_swap 4, chain 4, hint_swap 3,
epilogue 1, fractional 1; `check-witness` also rejects all 83; the adapter's honest unit and honest 2-unit VU are
accepted). Positives: 64/64 and 4096/4096 `vu-k1536` accepted with the recorded words (table).

**Transcript vectors for lane a-gpu:** `fixtures/babybear-transcripts/` (`transcript_vu4.json`, `transcript_vu64.json`,
run r20260922-070527-a263; `README.md` there gives the byte-exact conventions: 4-byte LE base elements, `c_0`-first
degree-6 coordinates, SHA-256 state chaining, the squeeze counter, the preamble, the message order, Merkle layout).

## Checker v2 (vy-cpu2, 12 threads, PCS mode, `--variant v2`; lane a-v2, 2026-09-22)

The same prover on the checker-min lane's lookup-centred arithmetisation of the SAME relation
(`backends/numerical/reports/checker_min.md`, `verity_numerical.checker.v2`): one `T_OP` lookup per BF16 word,
`ALIGN4` + `SHIFT` tables instead of the (q, r) division hints, `LEAD` / `LEADNORM` / `TNORM` instead of the leading-bit
/ normalise / state-out chain; the VU chains the *state* (s, e, M, z) and the group saturation flags between units and
packs once, in the epilogue. Exporter `gkr_export/v2.py` (`--variant v2` on `gkr_export.vu`); per unit **264 columns /
290 wires / 26 products / 86 lookups into 10 tables, multiplicative depth 1** (v1: 468 / 826 / 358 / 374 into 9, depth
4). The 5.14M fixed table rows are *implicit* (`itable` lines; `circuit.rs::materialise`, dense index) — only their
multiplicities are committed, per proof. Chain: 8 links, **583 linkage constraints per VU** (v1: 98), including
`ovf = 0` on every non-final unit. `PROTOCOL.md` §14. Security **2^-127.71** at B = 64 and 4096
(`account_candidate_a` on the exported shape; the `SHIFT` / `T_OP` logUp identities are the worst lookup terms at
2^-169 / 2^-166 — the budget is still the commitment hash and the Ligero query test).

~~~sh
cargo test --release                                                       # 14 tests incl. the v2 fixtures / itable materialisation
uv run --package verity-numerical python -m verity_numerical.gkr_export.v2 negatives --out DIR   # 30 unit + 4 VU mutations
VARIANT=v2 GATE=1      bash backends/gkr/run_measure_vu.sh                 # cargo test; generated v2 unit + VU negatives
VARIANT=v2             bash backends/gkr/run_measure_vu.sh 4096            # export (--variant v2) + prove + verify
VARIANT=v2 NEGATIVES=1 bash backends/gkr/run_measure_vu.sh                 # vu-k1536-neg
~~~

**v1 vs v2, Goldilocks^3, same box (shared with the latency and b-lookup lanes, load ≈ 35–48 on 32 vCPU during every
run), 12 threads, all recorded words accepted** (v1 rows: the a-babybear lane's post-R<w>-fix control, commit 54e5da7;
v2: commits d30efa3 / bf81577, the same `gkr.rs` / `logup.rs` / `ligero.rs`):

| B | checker | run id | prove_s (commit / lookup / arith / open) | verify_s | proof bytes | rounds | depth | peak RSS | committed | s / VU | v2 / v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | v1 | r20260922-064620-cf4c | 5.79 (0.22 / 1.89 / 3.22 / 0.43) | 0.49 | 1,638,920 | 1974 | 234 | 587 MB | 2,959,871 | 0.0905 | — |
| 64 | v1 (re-run, d30efa3) | r20260922-075822-5076 | 5.39 (0.17 / 1.67 / 3.12 / 0.40) | 0.43 | 1,638,920 | 1974 | 234 | — | 2,959,871 | 0.0843 | — |
| 64 | **v2** | r20260922-080202-8e4d | **6.13** (0.57 / 3.94 / **0.31** / 1.29) | 1.35 | 3,087,392 | 2199 | 279 | 798 MB | 6,770,360 | 0.0958 | prove 1.06×, arith 0.10×, lookup 2.1×, committed 2.3× |
| 4096 | v1 | r20260922-065844-2b84 | 261.3 (9.1 / 63.8 / 160.1 / 26.1) | 26.1 | 69,693,032 | 3179 | 381 | 33.8 GB | 184,145,855 | 0.0638 | — |
| 4096 | v2 (overlapping the `vu-k1536-neg` run) | r20260922-080548-b67d | 57.0 (5.8 / 21.4 / 15.3 / 13.6) | 11.9 | 41,510,240 | 3063 | 328 | 13.4 GB | 109,021,880 | 0.0139 | prove 0.218× |
| 4096 | **v2 (headline)** | **r20260922-081106-6067** | **54.2** (5.6 / 20.2 / **15.6** / 11.9) | **12.2** | **41,510,240** | 3063 | **328** | **13.3 GB** | **109,021,880** | **0.0132** | **prove 0.207× (4.8×), verify 0.47×, bytes 0.60×, RSS 0.39×, committed 0.59×, depth 328 vs 381** |

(peak RSS from the harness's process sampler, `resources.jsonl`, 5-s samples.) `overhead.vs_native_peak` 1.34e9
(v1 6.48e9). Measured split ratios v2 / v1 at B = 4096: arith **0.098**, lookup **0.317**, commit + open **0.50**,
committed elements 0.59 — the arithmetic bucket collapsed by the modelled ~10× (26 products of depth 1 against 358 of
depth 4) and the lookup bucket by 3× despite 5.1M table leaves; the proof is now **37 % lookup, 29 % arith, 32 %
commit + open**. Verify 12.2 s is dominated by the table-side evaluation of the implicit tables (5.14M rows) and the
Ligero column re-encoding.

**Correctness gate** (all on vy-cpu2, binary of d30efa3): `cargo test --release` 14/14 (r20260922-075226-1f2e); the
generated v2 transition-unit negatives **30/30 rejected** (every hint, lookup-output and outgoing-state column of a unit
mutated by +1; 13 caught at the phase-2a assertion sumcheck, 17 at the affected table's logUp identity) and the
4 synthetic VU negatives **4/4 rejected** (`broken_chain`, `false_overflow_flag`, `public_wrong_word`,
`saturated_intermediate` — all at the linkage functional); `vu-k1536-neg` **52/52 rejected, 0 wrongly accepted**
(r20260922-080600-a365: 8 at witness generation — the non-finite / saturated operands are outside `T_OP`'s domain —
and 44 by the verifier at the phase-2a assertion sumcheck). The Z-level wrap cases of the batteries are what the
`+1` mutations and the `broken_chain` / `public_wrong_word` cases exercise on Goldilocks; the BabyBear-specific
limb cases do not apply (no BabyBear v2 instantiation: `state_decode` / `epilogue` need the verified limb split first).

**Table-commitment decision and small B.** The tables are implicit but their *multiplicities* are committed and their
rows are leaves of the fractional-sum trees, so every proof pays 5.14M elements + 5.14M leaves regardless of B. At
B = 64 that is 76 % of the committed elements and the reason v2 is *not* faster there (6.13 vs 5.79 s: lookup 3.9 vs
1.9 s, open 1.3 vs 0.4 s, while arith fell 3.2 → 0.3 s). Fitting prove_s = a + b·B to the two points: v2 ≈ 5.4 s +
0.0119 s/VU, v1 ≈ 1.7 s + 0.0634 s/VU, so **v2 wins from B ≈ 72** and is 4.8× at 4096; the fixed cost is < 10 % of the
proof from B ≈ 4000. The split-u `SHIFT` fallback (2 × 2^8·54 rows instead of 2^16·54) would cut the fixed cost by
~3.4M rows and matter only below B ≈ 300; it is not built.

**GPU re-price** (`explore.tensor`, `VERITY_CHECKER_VARIANT=v2`, `ARITHMETIC_DIAGNOSTIC`): scaling the v1 model's
arith / lookup / commit buckets by the measured CPU ratios (0.098 / 0.317 / 0.50) gives H100 at utilisation 0.172:
v1 model 0.505 s per 4096 batch (overhead 1.25e7) → v2 model 0.221 s (5.47e6) → **v2 re-priced 0.150 s (3.72e6)**,
with the commitment (encode + hash, 0.105 s) now 70 % of the GPU time; BabyBear^6 re-priced 0.103 s (2.54e6). The model's
own v2 row is conservative on the arith bucket (it prices the census's 33 products at depth 3; the export has 26 at
depth 1).

## Batched logUp over the table union (vy-cpu, 12 threads, PCS mode, `--variant v2 --logup union`; lane a-logup-union, 2026-09-22)

One fractional-GKR instance over all twelve v2 tables instead of one per table: every query and every row is tagged
by its table, \(\phi(w,t)=\sum_k w_k\beta^k+\gamma\,\mathrm{tag}(t)\), one \((z,\beta,\gamma)\) message drawn after
the commitment. `PROTOCOL.md` §15 (construction, soundness, depth, GPU shape); `gpu/HANDOFF_UNION.md` for the GPU
lanes. **A protocol change to the lookup argument only; relation, export, commitment and chain unchanged.** The
per-table path stays the default; `--logup union` / `VERITY_GKR_LOGUP=union` / `LOGUP=union|both` in
`run_measure_vu.sh` select the union (`LOGUP=both` runs both legs on the same binary, box and instances and writes
`result_per_table.json` + `result.json`). Security **2^-127.71 at both fields and both B, unchanged** (the union's
collision 2^-165.8 + identity 2^-166.8 + GKR 2^-182.0 terms replace the twelve per-table identities, lookup bucket
2^-165.2 → 2^-165.2; `account_candidate_a(lookup_mode="union")`, `V2_EXPORT_TABLES`).

~~~sh
VARIANT=v2 GATE=1 LOGUP=both      bash backends/gkr/run_measure_vu.sh        # cargo test (17, both modes); 30 + 4 v2 negatives per mode
VARIANT=v2 NEGATIVES=1 LOGUP=both bash backends/gkr/run_measure_vu.sh        # vu-k1536-neg in both modes
VARIANT=v2 LOGUP=both             bash backends/gkr/run_measure_vu.sh 4096   # per-table then union, one run dir
~~~

**Correctness gate** (vy-cpu, commit 2f547db): `cargo test --release` 17/17 (r20260922-093758-9aac; honest units in
both modes, every exported negative rejected in both modes, a per-table proof rejected by a union verifier and vice
versa, the redteam-3 `cross_table` and `forged_epilogue_hint` fixtures rejected); v2 unit negatives **30/30** and VU
negatives **4/4** rejected in per-table AND union mode (same run); `vu-k1536-neg` **52/52** in both modes
(r20260922-095946-5638: 8 at witness generation, 44/44 by the verifier, per mode).

**Per-table vs union, same run, same binary, all VUs accepted:**

| B | lookup | run id | prove_s (commit / lookup / arith / open) | verify_s | proof bytes | coins | depth (lookup chain) | peak RSS | committed | s / VU | union / per-table |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | per-table | r20260922-101533-20c4 | 7.37 (0.94 / 4.41 / 0.29 / 1.72) | 1.76 | 3,152,048 | 2525 | 257 (254) | 1.9 GB | 6,868,664 | 0.1152 | — |
| 64 | **union** | r20260922-101533-20c4 | **7.09** (1.00 / **3.96** / 0.32 / 1.79) | 1.62 | 2,992,520 | **642** | 280 (277) | 1.9 GB | 6,868,664 | 0.1107 | total **0.96×**, lookup 0.90×, bytes 0.95×, depth +23 |
| 4096 | per-table (run 1) | r20260922-102013-2931 | 62.93 (9.00 / 25.43 / 13.89 / 14.17) | 17.19 | 41,574,896 | 3393 | 304 (301) | 13.4 GB | 109,120,184 | 0.0154 | — |
| 4096 | union (run 1) | r20260922-102013-2931 | 70.49 (6.62 / 30.16 / 17.14 / 16.01) | 15.31 | 41,347,304 | 761 | 355 (352) | 13.4 GB | 109,120,184 | 0.0172 | total 1.12×, lookup 1.19× |
| 4096 | per-table (run 2) | r20260922-102556-cdcc | 67.36 (9.53 / 28.29 / 15.20 / 13.80) | 14.87 | 41,574,896 | 3393 | 304 (301) | 13.5 GB | 109,120,184 | 0.0164 | — |
| 4096 | **union (run 2)** | r20260922-102556-cdcc | **70.08** (8.49 / **29.45** / 14.82 / 16.80) | 15.17 | 41,347,304 | **761** | 355 (352) | 13.5 GB | 109,120,184 | 0.0171 | total **1.04×**, lookup 1.04×; two-run means **1.08× / 1.11×**, bytes 0.995×, depth +51 |

Reading: on the CPU the union is a wash — slightly faster at B = 64 (twelve trees' fixed costs dominate there),
slightly slower at B = 4096, where its one tree of \(2^{26}\) leaves (39.1M live, 42 % padding) does 1.25× the leaf
work of the twelve smaller padded trees (\(2^{24}\)·2 + \(2^{23}\) + \(2^{21}\)·4 + \(2^{20}\)·3 + \(2^{17}\) +
\(2^{16}\) = 53.7M). Buckets with identical work (arith, commit) vary ±20 % between the two runs on this box, so the
B = 4096 difference is within noise; the **coins** (3393 → 761) and the **depth** (+51 trips: one 26-level chain vs
the deepest 24-level per-table chain) are exact. Proof bytes barely move (the opening dominates). The union's case is
the GPU's per-instance latency (`reports/a_kernels_h100_v2.md`: ~14 ms × 10 instances), not CPU throughput —
`gpu/HANDOFF_UNION.md`. Not a `--breakthrough` (no > 1.5× change). vy-cpu is slower than vy-cpu2 (a-v2's 54.2 s
per-table headline): the two legs here are the like-for-like comparison, not the absolute numbers.
