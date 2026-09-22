# b-ligero: a specialised Ligero-style direct-constraint prover for the transition unit (torch, GPU)

Track B, lane `b-ligero`. Design: `PROTOCOL.md`. Proof class **NON_ZK_PROOF_DIAGNOSTIC** (no masking; hash and
challenge expander are host placeholders; the relation is WP2's checker, nothing dropped). Every number below was
measured on `vy-g3` (NVIDIA L40S 48 GB, torch cu124) on the frozen instance set `fixtures/bench-instances/v1`.

## What it is

* The unit's witness is one column of `W` (m rows x l units): 99 pinned decode rows, 92 hint rows (the checker's
  `gen_step` hints), 170 product rows, **2401 bit rows** (every range check as an explicit decomposition), 610
  one-hot selector rows (the four fixed tables `POW`, `ALIGN`, `LEAD`, `NORM` as one-hot lookups over their
  distinct rows) -- **3372 rows per unit**, 338 linear + 3246 quadratic constraints (the census's 358 mul / 137 lin /
  306 lookups, with the lookups paid as bits and selectors).
* BabyBear, rate-1/4 Reed-Solomon (`[16384, 4096]`) by a batched radix-2 int64 NTT in torch, Blake3 Merkle over
  the codeword columns on the host, Ligero's three tests (interleaved proximity; linear constraints with the
  verifier encoding the public right-hand side itself, so nothing is sent; quadratic constraints as a batched
  Hadamard identity with a quotient by `Z_H`), `t = 189` opened columns, 6 BabyBear coordinates per challenge:
  soundness `2^-128.1` by `security.accounting.ligero_error`. `rounds.sequential_depth = 2`.
* A compile-time **integer audit** (`compile.py`): every linear identity's integer interval must lie in `(-p, p)`,
  every product's in `(-p, p)` -- or be a whitelisted theorem of the relation (three, listed in `unit.py`) -- so
  each field identity is an integer identity. Where the checker's words do not fit BabyBear (`2^32 > p`) the
  predicates are restated on the components (PROTOCOL.md section 5; the accepted set is unchanged).
* **Chain** (`chain.py`, `vu.py`): 96 units per VU in consecutive columns; the accumulator's components become hint
  rows decoded in-field; the epilogue (RN-even cast) is constrained per column; the cross-column linkage
  `c_in[j+1] = c_out[j]`, `c_0 = 0` and `y16 = claimed` are Ligero's *general* linear test (position-dependent
  coefficients interpolated per row; one polynomial `q` of degree `< 2l - 1` sent; no rotation, no extra openings).
  3516 rows per unit, 347 linear + 3387 quadratic + the chain system.

## Runs

~~~
# on vy-g3, /workspace/bl/src, PYTHONPATH=packages/verity/src:backends/numerical/python:.
python -m backends.direct.ligero.run gate     --root fixtures/bench-instances/v1 --device cuda --vus 64   # results/gate_unit.log
python -m backends.direct.ligero.run bench    --root fixtures/bench-instances/v1 --device cuda --reps 5 --out results/result_unit_b4096.json
python -m backends.direct.ligero.run gate-vu  --root fixtures/bench-instances/v1 --device cuda --vus 64   # results/gate_vu.json
python -m backends.direct.ligero.run bench-vu --root fixtures/bench-instances/v1 --device cuda --reps 5 --out results/result_vu_42.json
python -m backends.direct.ligero.encoder_bench                                                          # NTT vs INT8 matmul DFT
# laptop (CPU, < 30 s): the same commands with --device cpu --batch 256 --vus 1
~~~

## Correctness gate (vy-g3)

| check | result |
|---|---|
| `tu-k16`: all 4096 units in one proof (l = 4096), witness from `gen_step`, prover self-check on | accept |
| `vu-k1536`: the 6144 units of 64 VUs (every recorded intermediate accumulator as a unit's public output) | accept |
| `vu-k1536`: 64 VUs as chains (42 + 22 per proof), recorded accumulators, claimed BF16 words | accept |
| `vu-k1536-neg`: 52 negatives as unit claims (`wrong` -> linear test; `reject` -> operand domain check) | 52/52 rejected |
| `vu-k1536-neg`: 52 negatives as VU claims (`wrong` -> chain test; `reject` -> domain check / foreign proof) | 52/52 rejected |
| mutated witnesses, unit: one row per (gadget scope, row class) changed in one column before commitment | 243/243 rejected |
| mutated witnesses, chain: 249 rows + broken link `c.f[40] != y.f[39]`, `c_0 != 0` at a start, flipped `y16` | 252/252 rejected |

## Measurements (CUDA events / synchronised host clock, warm, median of 5; `results/*.json` per `bench.contract`)

| | unit proof, B = 4096 units | VU proof, 42 VUs (l = 4096) |
|---|---|---|
| prover total | **3.56 s** (0.87 ms/unit, 83 ms per 96 units) | **3.82 s** (**91 ms/VU**) |
| of which: hints (`gen_step`, host Python) | 2.59 s | 2.72 s |
| witness (torch program: derived rows) | 0.14 s | 0.15 s |
| encode (int64 NTT, 3372/3516 x 4096 -> 16384) | 0.12 s | 0.12 s |
| transfer of U to host | 0.37 s | 0.40 s |
| Merkle (host Blake3, placeholder) | 0.28 s | 0.36 s |
| tests (w, quotient h, chain q) | 0.06 s | 0.07 s |
| openings | 0.002 s | 0.002 s |
| verifier | 0.12 s | 0.034 s (NTT evaluation instead of Horner) |
| proof bytes | 2.83 MB (2.55 MB opened columns) | 3.14 MB (75 KB/VU) |
| committed | 3372 x 16384 x 4 B = 221 MB | 230 MB (5.5 MB/VU) |
| peak device memory | 2.83 GB | 2.95 GB |
| soundness (accounting) | 2^-128.1 (t = 189, D = 6) | 2^-128.1 |
| `rounds.sequential_depth` | 2 | 2 |
| overhead vs native 9.85 ps/VU | 8.5e9 (scope unit, x96) | **9.2e9** (scope vu) |

GPU-side prover alone (everything but the host hint generator): 0.97 s per 4096 units = **0.24 ms/unit**, 1.10 s
per 42 VUs = **26 ms/VU**; the L40S is 65% idle inside that (transfer + host hashing).

## Where the cost model (`tensor.md`, row B/BabyBear) was wrong

* **Committed bytes**: the model prices B at 1.16 MB/VU; the direct-constraint unit commits 5.5 MB/VU, because the
  306 lookups per unit became 2401 bit rows + 610 selector rows (3011 of 3372 rows). The AIR-with-lookups B and the
  direct-constraint B are different objects; the census's "358 mul / 137 lin / 306 lookups" undercounts the
  committed width of the latter by 4.7x. A lookup argument (LogUp with helper columns) would bring m to ~700.
* **Hashing is the wall the model said it was, plus the transfer it did not price**: 0.37 s to move 221 MB to the
  host and 0.28 s to hash it (790 MB/s single-threaded Blake3) are 65% of the GPU-side prover; the model's
  "B is a hashing problem" is right, and the fix (GPU Merkle, `backends/shared/hash_gpu`) removes the transfer too.
* **Encoding is not the bottleneck at this shape**: 0.12 s (3% of the prover) with the int64 NTT; the INT8 limb
  matmul DFT reaches 139 TOPS on the tensor cores (52 ms for the one-stage dense product) but its limb recombination
  in torch costs 220 ms -- it needs a fused kernel to pay (`encoder_bench.py`, PROTOCOL.md section 8).
* **Hint generation** is priced at 0.5 ms per batch ("trivially parallel, unmeasured"); on the host it is 2.6 s per
  4096 units -- 71% of the measured prover. It is the checker's own Python `gen_step`, used deliberately as the
  reference; a torch hint generator is the obvious next step and turns the 91 ms/VU into ~30 ms/VU.
* **Depth**: 2, not 3 (no lookup helper message).

## Negatives and labels

Every number: `NON_ZK_PROOF_DIAGNOSTIC`, authentication excluded, hardware NVIDIA L40S, security `2^-128` by the
accounting (hash and SHAKE challenge expander are placeholders), scope `unit` numbers omit chain linkage / epilogue
(the `vu` row has them), ZK masking omitted (its cost: PROTOCOL.md section 8b -- at `2^-128` it doubles the codeword
length, so encoding / transfer / hashing double; +0.5 s on the 1.0 s GPU-side prover). No predicate of WP2's
`check_step_expr` / `check_vu_expr` is dropped; the restatements are listed in PROTOCOL.md section 5 and 7b.

## Files

`compile.py` (Expr, LigeroCtx, the integer audit, lookups as one-hot selectors) . `unit.py` (the unit and chain
systems; public vectors) . `field.py` (BabyBear, NTT, coset encoding) . `merkle.py` (Blake3/SHA-256 placeholder) .
`witness.py` (W from the program; sparse constraint tables) . `chain.py` (layout, the chain test's coefficients) .
`protocol.py` (prove / verify) . `vu.py` (chain gate / bench) . `run.py` (CLI) . `encoder_bench.py` .
`results/` (gate logs, result envelopes). Ledger: `backends/reports/ledger/b-ligero.jsonl`, plot
`backends/reports/plots/overhead_b-ligero.png`.
