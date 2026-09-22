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

## What the transcript must not let the prover re-supply

- [ ] `magW` (never range-checked in the relation; `forge_split`) -- must be derived in-circuit from committed
      `normB`, `r_n`, not accepted from the prover after any challenge.
- [ ] `aligned`, `mag24`, `L` over a field below `2^{2A}`, `2^{24+K}`, `2^{2B}` respectively (`forge_align`,
      `forge_final`, `forge_leading_bit`) -- the field size in the protocol document must be stated and checked against
      `forge.thresholds` / `forge.max_products` for the pipeline in use.
- [ ] `neg_t` when `total = 0` (free column) -- confirm the protocol does not expose it (it is unobservable in the
      relation but would be a distinguishing bit in a transcript).
