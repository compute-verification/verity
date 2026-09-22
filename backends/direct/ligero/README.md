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
  the codeword columns on the GPU (`backends/shared/hash_gpu`; host Blake3 fallback), Ligero's three tests (interleaved proximity; linear constraints with the
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
python -m backends.direct.ligero.run bench-vu --root fixtures/bench-instances/v1 --device cuda --reps 5 --batch 8192 --vus 85 --out results/result_vu_85.json
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

## Measurements (synchronised device clock, warm, median of 5; `results/*.json` per `bench.contract`)

Final state (torch hint generator, GPU Blake3 Merkle from `backends/shared/hash_gpu`, `torch.compile`d NTT butterfly
and constraint kernels, CUDA-graph replay of the witness program):

| | unit proof, B = 4096 units | VU proof, 42 VUs (l = 4096) | VU proof, 85 VUs (l = 8192) |
|---|---|---|---|
| **prover total** | **0.095 s** (**23.2 us/unit**, 2.2 ms per 96 units) | **0.108 s** (**2.57 ms/VU**) | **0.195 s** (**2.30 ms/VU**) |
| **prover total on the A100** (anchor-a100 lane, NVIDIA A100-SXM4-80GB, run `r20260922-081113-f56e`, same commands, 7 reps, verifier accepted every rep; gate-vu 52/52 negatives + 252/252 mutations rejected) | **0.072 s** (**17.6 us/unit**; 1.32x the L40S) | **0.084 s** (**1.99 ms/VU**; 1.29x) | **0.151 s** (**1.78 ms/VU** = overhead 1.81e8 vs the same card's BF16 peak; 1.29x) -- encode 0.044 s, tests 0.048, hints 0.036, witness 0.016; `backends/numerical/reports/anchor_a100.md` section 7 |
| hints (torch `gen_step` as one CUDA graph, + the public decodes; cross-checked equal to the checker's) | 0.017 s | 0.017 s | 0.031 s |
| witness (torch program as one CUDA graph: products, bits, selectors) | 0.008 s | 0.008 s | 0.009 s |
| encode (int64 NTT, compiled butterfly, 3372/3516 x l -> 4l) | 0.031 s | 0.032 s | 0.069 s |
| int32 view of U (nothing leaves the device) | 0.001 s | 0.001 s | 0.002 s |
| Merkle (Blake3 on the GPU, 221 / 230 / 461 MB) | 0.001 s | 0.001 s | 0.002 s |
| tests (w, quotient h, chain q; compiled combine / sparse-apply) | 0.034 s | 0.044 s | 0.079 s |
| openings (m x t columns to the host, paths) | 0.003 s | 0.003 s | 0.004 s |
| verifier | 0.030 s | 0.039 s | 0.054 s |
| proof bytes | 2.83 MB (2.55 MB opened columns) | 3.14 MB (75 KB/VU) | 3.54 MB (42 KB/VU) |
| committed | 3372 x 16384 x 4 B = 221 MB | 230 MB (5.5 MB/VU) | 461 MB |
| peak device memory | 3.18 GB | 3.31 GB | 6.61 GB |
| soundness (accounting) | 2^-128.1 (t = 189, D = 6) | 2^-128.1 | 2^-128.1 |
| `rounds.sequential_depth` | 2 | 2 | 2 |
| **overhead vs native 9.85 ps/VU** | 2.3e8 (scope unit, x96) | **2.6e8** (scope vu) | **2.3e8** (scope vu) |

How it got there (ledger `b-ligero.jsonl`, all on the L40S):

| step | unit prover (4096) | VU prover (42) | ms/VU |
|---|---|---|---|
| first complete prover: Python `gen_step` hints (2.6 s) + host Blake3 | 3.56 s | 3.82 s | 91 |
| torch hint generator (bit-exact with `gen_step`; 2.6 s -> 0.02 s) | 1.06 s | 1.07 s | 25 |
| GPU Merkle (`hash_gpu`): transfer 0.40 s + hashing 0.30 s -> 2 ms | 0.345 s | 0.367 s | 8.7 |
| `torch.compile` NTT butterfly (7 passes/stage -> 1; 93 -> 21 ms per NTT) | 0.25 s | 0.27 s | 6.4 |
| compiled combine / sparse-apply (tests 0.062 -> 0.034 s) | 0.19 s | 0.20 s | 4.8 |
| CUDA-graph replay of the witness program (~2500 launches; 55 -> 8 ms) | 0.101 s | 0.110 s | 2.6 |
| CUDA-graph replay of the hint generator (23 -> 17 ms) | 0.095 s | 0.108 s | 2.57 |

What is left is the NTT (33%), the tests (36% -- a sparse matmul of the constraint tables against `W`, still int64)
and the hints + public decodes (18%: numpy -> device copies of the operands and ~200 small kernels with a few domain
checks that synchronise). Both big items are int64 elementwise work; the tensor-core DFT (section 8) is the next
step for the NTT, an int32 sparse matmul for the tests.

## Negatives and labels

Every number: `NON_ZK_PROOF_DIAGNOSTIC`, authentication excluded, hardware NVIDIA L40S, security `2^-128` by the
accounting (hash and SHAKE challenge expander are placeholders), scope `unit` numbers omit chain linkage / epilogue
(the `vu` row has them), ZK masking omitted (its cost: PROTOCOL.md section 8b -- at `2^-128` it doubles the codeword
length, so encoding and hashing double: +0.03 s on the 0.095 s prover, ~+33%). No predicate of WP2's
`check_step_expr` / `check_vu_expr` is dropped; the restatements are listed in PROTOCOL.md section 5 and 7b.

## v2 shape, end to end (`v2/`, lane b-e2e-v2, vy-g2 RTX 4090, 2026-09-22)

The checker-v2 relation (`checker/v2/`, `gkr_export.v2`: 273 columns / 33 products / 37 asserts / 93 lookups into
13 tables per unit, the `(s, e, M, z)` state chain, limb pack + epilogue rows) as a Ligero proof with LogUp tables,
\(F_c = \text{BabyBear}^6\) challenges and a torch-vectorised verifier; PROTOCOL.md section 16.  Layout: one
Ligero column per unit, one row per constraint-system column; the table side (SHIFT 3.5 M + TNORM 1 M entries
as-is) is 60-75% of the rows.  `NON_ZK_PROOF_DIAGNOSTIC` with the `INCOMPLETE_CHECKER_DIAGNOSTIC` caveat (the
limb rows are a port of b-air-v2's z3-UNSAT design; the Python columns were not z3-checked).

~~~sh
# pod (commit first); instances are built once into /workspace/bi/v1
uv run --frozen research run --on vy-g2 --source . --project verity --stage b_e2e_v2.gate  --require-result -- bash backends/direct/ligero/v2/pod_gate.sh
uv run --frozen research run --on vy-g2 --source . --project verity --stage b_e2e_v2.bench --require-result --env LS="16384 32768" -- bash backends/direct/ligero/v2/pod_bench.sh
# laptop: the CPU toy of the whole protocol
python -m pytest backends/direct/ligero/v2/tests/test_toy.py
~~~

Gate (`r20260922-091001-68f7`): 8 honest `vu-k1536` VUs accepted with the recorded words (every packed intermediate
= the recorded accumulator; prover self-check: every constraint satisfied); `vu-k1536-neg` 52/52 + flipped word
rejected; 39/39 unit + chain mutations rejected; battery `epilogue` 2/2 rejected (`chain` class vacuous under v2).

End to end, 4 096 VUs per batch as N sub-batch proofs (`bench`, all sub-batches proved and verified):

| l (units/proof) | VUs/proof | N | prover s/proof | **ms/VU** | s per 4 096 VUs | overhead | verifier s/proof | proof MB | rows m | peak GiB | union security | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 384 | 170 | 25 | 0.728 | **4.28** | 17.5 | 4.3e8 | 0.26 | 4.5 | 3 139 | 8.5 | 2^-123.5 (t = 189 per proof; t = 194 meets 2^-128) | r20260922-094315-8be5 |
| 32 768 | 341 | 13 | 0.888 | **2.60** | 10.7 | 2.6e8 | 0.29 | 5.7 | 2 019 | 11.8 | 2^-124.5 | r20260922-094315-8be5 |
| 65 536 | 682 | 7 | see `results/bench_l65536_*.json` | | | | | | | | | r20260922-094956-5453 |

Host hint generation (`gen_unit`, pure Python, pool of 95 processes) is 5.4 / 8.1 s per sub-batch and is reported
separately from the device clock.  Before the GPU Blake3 Merkle was importable on the pod (venv without cupy /
blake3) the same proof took 5.8 s (host SHA-256 over 65 536 columns); before b-v2's blocked Montgomery inverter
replaced the torch Frobenius inverse, 1.66 s.

Buckets, l = 32 768 (ms per proof of 341 VUs) against b-v2's synthetic batch (ms per 4 096 VUs, one fused proof):

| bucket | real (this prover, per proof) | real per 4 096 VUs (x 13) | b-v2 synthetic | what the delta is |
|---|---|---|---|---|
| witness -> device (columns, products, multiplicities) | 206 | 2 680 | -- (hints not in b-v2) | host numpy -> int64 tableau; multiplicity bincount |
| encode (int64 torch NTT, 2 tableaux) | 30 + 104 | 1 740 | 16.2 (radix-4 fused encode + Blake3) | eager NTT, 8 B/element; `commit_stream` not wired (systematic-domain encoder vs this protocol's coset) |
| Merkle (GPU Blake3) | 2 + 5 | 91 | (in 16.2) | -- |
| challenges (SHAKE expand, table image at beta, pole rule) | 44 | 575 | 2.7 modelled | host SHAKE + 13 table evaluations at beta |
| helpers (F_c inversions, b-v2 blocked kernel) | 113 | 1 470 | 5.2 | the kernel is ~20 ms; the rest is `_lins_eval` + F_c products in eager torch |
| IRS + sums | 84 | 1 090 | 2.10 | eager r^T W |
| quadratic (F_t products + F_c helper triples) | 279 | 3 620 | 5.4 | 6 x 6 coordinate products streamed in eager torch |
| chain | 8 | 105 | (in structured linear 8.29) | -- |
| openings | 13 | 170 | 2.5 | -- |
| **total** | **888** | **11 540** | **42.4** | **270x**: eager torch over 2 019 x 131 072 int64 vs fused int32 kernels over 104 217 x 16 384 |

## Files

`compile.py` (Expr, LigeroCtx, the integer audit, lookups as one-hot selectors) . `unit.py` (the unit and chain
systems; public vectors) . `field.py` (BabyBear, NTT, coset encoding) . `merkle.py` (GPU Blake3 via `hash_gpu`;
host Blake3/SHA-256 fallback) . `hints_torch.py` (vectorised `gen_step`) .
`witness.py` (W from the program; sparse constraint tables) . `chain.py` (layout, the chain test's coefficients) .
`protocol.py` (prove / verify) . `vu.py` (chain gate / bench) . `run.py` (CLI) . `encoder_bench.py` .
`results/` (gate logs, result envelopes). Ledger: `backends/numerical/reports/ledger/b-ligero.jsonl`, plot
`backends/numerical/reports/plots/overhead_b-ligero.png`.
