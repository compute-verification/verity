# Transcript ordering / leakage checklist

Run against every protocol document (prover-verifier message schedule, commitment layout, challenge derivation) as it
lands.  Each box is one attack; a box is closed by a pointer into the document (section, message index) or by a
reproducer under `redteam/`.  An unchecked box is an open question, not a pass.

## Ordering

- [ ] **Challenge before its commitment.** For every challenge `c_j`, list the commitments it is derived from; confirm
      each is absorbed into the transcript strictly before `c_j` is squeezed.  Attack: pick the witness after seeing
      `c_j` (Fiat-Shamir on a partial transcript).
- [ ] **Multiplicities fixed after the challenge.** For every lookup / logup argument, confirm the multiplicity column is
      committed before the challenge `beta` / `gamma` that the sum `sum 1/(beta - t_i)` depends on.  Attack: choose
      multiplicities as a function of `beta` so the rational identity holds for a table the witness does not use.
- [ ] **Range chunks after the challenge.** The 16-bit chunk decomposition of every range-checked column (`aligned`,
      `rinv`, `rsinv`, `mag_t`, `r_n`, `magW` if it is added) is committed before any challenge that batches the
      range lookups.
- [ ] **Quotient / opening order.** The quotient polynomial commitment precedes the evaluation point; the evaluation
      point precedes every claimed opening.  Attack: an opening chosen to fit the point.
- [ ] **Chained steps.** In the K=1536 chain, group `i+1` consumes the *committed* output of group `i`, not a value
      re-supplied by the prover after a challenge (the forged `magW` of `forge_split` becomes a free re-entry point
      otherwise).
- [ ] **Public inputs bound first.** The BF16 words, the accumulator and the claimed output are absorbed before the
      first challenge; the "E* attained" chain and every table are fixed by the public parameters, not by the prover.

## Restart / grinding

- [ ] **Restart on an unfavourable challenge.** Is there any prover-controlled nonce, salt, blinding or padding that
      enters the transcript before a challenge?  If so, count the grinding budget: `bits(nonce)` retries at cost of one
      commitment each.  Attack: grind until the batching challenge makes a forged range chunk cancel.
- [ ] **Soundness error per challenge vs. grinding.** For each challenge with soundness error `eps`, state the
      effective error `eps * 2^{grind bits}` and check it against the target level.
- [ ] **Verifier-side randomness reuse.** The same challenge is not reused across two independent arguments
      (lookup batching and range batching share `beta` only if the analysis covers the joint sum).

## Mask / blinding reuse

- [ ] **Mask reuse across contexts.** Every zero-knowledge blinding polynomial / mask is fresh per commitment and per
      proof; no mask is derived from a value that appears elsewhere in the transcript.
- [ ] **Mask reuse across the chain.** In the K=1536 chain the per-step masks are independent; two steps sharing a mask
      leak the difference of the masked columns (the signed totals, hence the products).
- [ ] **Deterministic masks from the witness.** A mask derived by hashing the witness leaks a witness-equality oracle
      across proofs on the same inputs.

## Message shapes depending on private values

- [ ] **Exponent-dependent shapes.** Number, size or ordering of messages (commitment count, chunk count, table rows
      touched) does not depend on `E*`, `d_i`, `L`, `e_out`, `ovf`.  Attack: measure transcript length / row count to
      read the exponents.
- [ ] **Selector-dependent shapes.** Which term attains the maximum, which `T_pow` / `T_exp` row is hit, whether the
      `ALIGN_TABLE` or `ALIGN_DIV` path is taken, is not visible from the message schedule or from the multiplicity
      column being non-zero at a row.
- [ ] **Branch-dependent shapes.** Special-value branches (`nan`, `inf`, zero total, `z_out`, `ovf_eff`) produce the
      same message sequence as the finite branch.
- [ ] **Multiplicities as a side channel.** The committed multiplicity vector of every table lookup is hidden (masked or
      inside a ZK commitment); a public multiplicity vector reveals the histogram of `d_i`, `L`, `e_out` over the step.
- [ ] **Timing / size of the prover's messages.** Proof size and commitment sizes are constant over the finite domain.

## Batch partitioning by private data

- [ ] **Partitioning by private data.** Any batching of rows / lookups / range checks into sub-arguments is by a
      public rule (row index, group index), never by a private value (exponent class, sign, zero flag).
- [ ] **Sorted / permuted columns.** If a permutation argument sorts by a private column, the sorted order is not
      revealed (the sorted column is committed, not sent).
- [ ] **Table-size partitioning.** Splitting `T_pow` (512 rows) or `T_exp` (512 rows) into sub-tables by row range
      partitions the lookups by `d_i` / `e_out`; confirm the split is not observable.
- [ ] **Sparse-lookup shortcuts.** A prover-side optimisation that omits lookups for rows with zero multiplicity
      changes the message shape with the private data.

## Review items for the build lanes' first commits (wave 2: from the WP2 checker contracts)

Both candidates implement `checker/gadgets.py` `contract()`s as composed by `step.py`.  Wave 2 found no accepted
deviation from the reference over the integers or over Goldilocks (`notes-asset:campaigns/r20-proof/assets/red-team-checker/reports/redteam_checker.json`: mutation /
coordinated on every hint column, z3 per gadget with Boolean case splits); every SAT is a field below the checker's
F1-F4 thresholds.  What an implementation can still get wrong is the *realisation*.  Check, per commit:

### In a GKR layer wiring (A-GKR) and in an AIR / trace (B-Ligero) alike

- [ ] **Every `rng` of the contracts is present, on the canonical representative.**  Diff the layer's / AIR's range
      queries against `contract().hints` + `contract().checked` (the `checked` expressions: `t-nz`, `M+z-1`, `u+z-1`,
      `d[i]`, `d_floor`, `r*ipow`, `q*pow`, `r*isr`, `q*sr`, `v`, `v+z-1`, `(v-lo)*ispan`, `clamp_proof`, `r*idiv`,
      `ovf_proof`, `hi+carry`).  A dropped one is a SAT (`checker/search.py::integer_queries`, `drop=`).  Signed ranges
      (`E`, `d`, `k`, `cand`, `e1`) are checked as `x - lo` in `[0, hi - lo)`, never as a signed comparison in the field.
- [ ] **Wrap thresholds of the trace field.**  With the product realisation of `lt_pow`, `p > 2^60` is required (F1:
      `leading_bit` is `(v - lo) * ispan < 2^30`; SAT at `p40`, reproducer `z3_checker.gadget_split(leading_bit, P40)`);
      `p > 2^32` for words as single elements (F4: `fp32_decode`, `pack`, `epilogue` SAT at Mersenne-31); `p >= 2^31 - 1`
      for `signed_sum` (now proved UNSAT at exactly `2^31 - 1`); `p > 2^31 - 1` for `normalise`'s `div M' + r` (F3).
      Goldilocks passes all four; any smaller trace field must switch `lt_pow` to a direct (lookup or bit) bound and split
      the words.  Confirm the protocol document states `p` and the check `p > 2^60` explicitly.
- [ ] **The tight quotient bounds (`quot`) are realised, not only `rem`.**  Over the integers they are implied and a
      compiler may drop them as "redundant"; over any field they are load-bearing (F2, `checker/search.py::modular_queries`).
- [ ] **Table multiplicities.**  `POW` (512 rows) is queried 2x, `ALIGN` (512) 16x, `LEAD` (31) 2x, `NORM` (32) 2x, the
      16-bit range table 284x per unit (`security.census.lookups_by_table`).  Multiplicities committed before the lookup
      challenge; the *same* query count on every unit (no zero-multiplicity shortcut: the histogram of `d`, `l`, `k` is
      private).  A vector-valued lookup on `(key, cols...)` binds *every* column of the row: a row with only `pow` bound and
      `ipow` free re-opens F1.
- [ ] **The `magW` lesson: no prover-supplied value stands in for a computed expression.**  In the checker every gadget
      output (`m`, `e`, `u`, `E - cand`, `q * sl`, `S`, `v`, `e1`, `k`, the packed word) is an *expression* of committed
      columns.  An implementation that materialises one as a column must bind it by an identity to the expression, in the
      same commitment round as the hints -- a column bound only by a later range check or lookup is a re-entry point
      (wave 1's `magW`: 388/870 groups forged at Goldilocks).  Grep the AIR for columns without a defining identity.
- [ ] **Query wires on computed wires.**  140 of the 306 lookup queries per unit are on expressions (`lt_pow` products,
      `t - nz`, `M + z - 1`, `d = E - cand`, `k`, ...; `census.queries_on_computed_wires`).  GKR: the lookup argument
      reads a *committed* copy, so the checker must bind copy = wire (A-GKR `commit_query_wires`); check the
      binding is in the layer, not assumed.  AIR: each is a column with its defining constraint (`census_review`: +44
      chunk columns for the 22 checks wider than 16 bits, +69 scaled rows for the sub-16-bit checks on plain columns if the
      one 16-bit table is used).
- [ ] **Chaining of units.**  `step.py` passes unit `i`'s output *expression* as unit `i+1`'s `c`; a trace that stores
      `c_{i+1}` as a column needs `c_{i+1} = word_i` as one linear identity per unit (`census_review`) -- committed before
      any challenge.  `c_0 = 0` is a constant, not a column.  An infinite intermediate word must be rejected by unit
      `i+1`'s `fp32_decode` range `t < 255`, not by a separate flag.
- [ ] **Contract-level range gap (U4).**  `state_out.e_hat` reaches the floor (-132) when `z = 1`, below `pack.e1`'s
      formerly declared `[-126, ..)`; harmless only because `pack` multiplies the finite word by `(1 - z)`.  The
      contract now declares `pack.e1` as the range actually fed (`exp_lo`) and its docstring states the mask
      dependency (`checker/gadgets.py`, `Pack`).  Still check: an implementation that reads `fld = e1 + 126 + hidden`
      without the `(1 - z)` mask emits a wrapped word; the group-to-group exponent edge is contained only by the
      per-group ceiling `e_hi(g) + growth` (`z3_checker.containment`), which the contracts do not state.
- [ ] **Fiat-Shamir domain separation.**  Every squeezed challenge is bound to (protocol id, `K`, `B`, table ids, the
      unit index range of the sub-batch, the round index); the same transcript prefix must not yield the lookup challenge
      of table `T` and the range-batching challenge (checklist item "verifier-side randomness reuse").  The public
      inputs (`a`, `b`, `y16`, `K`) are absorbed before the first challenge; the tables by their identifiers, not their
      contents re-sent by the prover.  A-GKR (interactive, libzk-style) must state where the interactive protocol
      is made non-interactive, if at all, and that the AHIV22 extractor argument survives it.
- [ ] **Negatives suite coverage.**  Port `tests/checker/test_negative.py` (every hint mutated in the handoff's named
      ways) to the backend's witness layout and additionally include: each `drop=` case of `search.integer_queries`
      (one predicate removed must make the backend *accept* a forged witness -- the test that a bound is load-bearing);
      the F1 witness `leading_bit: v = 537133057, l = 19` and F2 `align_state: M = 0, d >= 25, q = 63, r = 2^25 - 1`
      replayed against the backend's range realisation (must reject); a coordinated two-column forgery
      (`harness.run_coordinated` on the backend's `check`) at the same case count as `notes-asset:campaigns/r20-proof/assets/red-team-checker/reports/redteam_checker.json` (1,909
      attempts, PASS).

## What the transcript must not let the prover re-supply

- [ ] `magW` (never range-checked in the relation; `forge_split`) -- must be derived in-circuit from committed
      `normB`, `r_n`, not accepted from the prover after any challenge.
- [ ] `aligned`, `mag24`, `L` over a field below `2^{2A}`, `2^{24+K}`, `2^{2B}` respectively (`forge_align`,
      `forge_final`, `forge_leading_bit`) -- the field size in the protocol document must be stated and checked against
      `forge.thresholds` / `forge.max_products` for the pipeline in use.
- [ ] `neg_t` when `total = 0` (free column) -- confirm the protocol does not expose it (it is unobservable in the
      relation but would be a distinguishing bit in a transcript).

## Statement and size binding (wave 3: from the red team's B forgeries, 2026-09-22, lane soundness-fixes)

- [ ] **The statement is bound, not inferred from the proof.** Every public value the verifier is asked to accept
      (claimed output words per VU, bound operands, B, K) is (a) absorbed into the challenger before the first
      challenge and (b) checked by the verifier against the OPENED trace: a public column is checked at ζ and ζ·g
      against the barycentric evaluation of the verifier's own copy of it (`direct/src/statement.rs::eval_pair`), or
      constrained via `AirBuilderWithPublicValues`. "The claimed words are committed cells" is NOT binding: a proof of
      a different satisfying trace (all-zero VUs) verifies as any statement (`b_zero_batch_as_any_statement`, F1).
      Pin: `direct/tests/chain.rs::statement_suite` (F1a/F1b), `BoundError::StatementColumn`.
- [ ] **Degree / size / shape come from the verifier's configuration, never from the proof.** `degree_bits`, the
      trace height, the number of VUs, the FRI parameters (blowup, queries, PoW bits), the field, the challenge
      extension degree and the AIR identity (a hash of the constraint program and layout) are fixed by the verifier
      from `(B, rows per VU, padding)` and its config, absorbed into the challenger, and any value the proof carries
      is compared, not used. Plonky3 uni-stark 0.4.3 reads `degree_bits` from the proof: a 2-VU (256-row) proof
      verified as 4096 VUs (`b_degree_bits_unbound`, F2). Pin: `statement_suite` (F2),
      `BoundError::DegreeBits{expected, got}`; header contents `statement.rs::header` (domain tag `verity-direct/stmt/v1`).
- [ ] **Every realisation of the same relation carries the binding.** A second prover/verifier pair for the same
      AIR (B's LogUp realisation, `direct/src/lookup_stark.rs`, a fork of uni-stark with two extra rounds) shipped
      with the pre-fix behaviour while the limb path was being fixed: no statement, `degree_bits` from the proof
      (`redteam-logup-b` cases `zero_batch_as_any_statement`, `degree_bits_from_proof`). Binding is a property of
      each verifier entry point, not of the AIR: enumerate the entry points (`verify`, `verify_lookup`, every
      `--mode`) and pin each. Fixed 2026-09-22 (lane `b-bind`, PROTOCOL.md 14.7): `prove_lookup_bound` /
      `verify_lookup_bound_bytes`, pin `direct/tests/lookup_statement.rs`. `verify_lookup` (unbound) remains as a
      diagnostic and is labelled `statement_binding.bound = false` in every result that uses it.
- [ ] **Preprocessed / public tables are part of the statement.** When the verifier recomputes a preprocessed
      commitment from a public table definition, the table definition (or its digest) is also absorbed into the
      challenger before the first challenge, so the two parties' challenges agree only if their tables do; the
      commitment comparison alone binds the prover's `P` but not the verifier's choice of tables to the transcript.
      Pin: `lookup_statement.rs` (substituted table row rejected both ways; `statement::table_digest`,
      `statement::lookup_header`, tag `verity-direct/chain-air/logup/v1`).
- [ ] **The trace height is a function of the statement AND the realisation.** The LogUp trace is at least the
      tables' height (2^17 rows for `|P| = 116 722`), so `Statement::rows()` differs between realisations for the
      same words (`Statement::min_rows`); a proof of the words at the wrong realisation's height is `DegreeBits`.
