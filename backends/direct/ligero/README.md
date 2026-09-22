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

## Files

`compile.py` (Expr, LigeroCtx, the integer audit, lookups as one-hot selectors) . `unit.py` (the unit and chain
systems; public vectors) . `field.py` (BabyBear, NTT, coset encoding) . `merkle.py` (GPU Blake3 via `hash_gpu`;
host Blake3/SHA-256 fallback) . `hints_torch.py` (vectorised `gen_step`) .
`witness.py` (W from the program; sparse constraint tables) . `chain.py` (layout, the chain test's coefficients) .
`protocol.py` (prove / verify) . `vu.py` (chain gate / bench) . `run.py` (CLI) . `encoder_bench.py` .
`results/` (gate logs, result envelopes). Ledger: `backends/numerical/reports/ledger/b-ligero.jsonl`, plot
`backends/numerical/reports/plots/overhead_b-ligero.png`.
