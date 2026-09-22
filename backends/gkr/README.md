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

## Layout

- `Cargo.toml`: standalone crate, **no external crates** (hand-rolled Goldilocks, cubic extension `X^3 − X − 1`, SHA-256).
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
