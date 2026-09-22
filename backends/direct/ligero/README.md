# b-ligero: a specialised Ligero-style direct-constraint prover for the transition unit (torch, GPU)

Track B, lanes `b-ligero` (first pass, sections below) and `b-ligero2` (second pass, next section: bound statement,
the 4096-VU decision workload, HVZK). Design: `PROTOCOL.md`. Proof class **NON_ZK_PROOF_DIAGNOSTIC** without `--zk`,
**COMPLETE_HVZK_BACKEND** with it (hash and challenge expander are host placeholders; the relation is WP2's checker,
nothing dropped). Every number below was measured on `vy-g3` (NVIDIA L40S 48 GB, torch cu124) on the frozen
instance set `fixtures/bench-instances/v1`.

## Second pass (lane b-ligero2, ledger `b-ligero2.jsonl`)

**1. Statement binding (SOUNDNESS fix, commit 302832f).** The red team's checklist (`redteam_vu.md` F1/F2/F19,
`TRANSCRIPT_CHECKLIST.md`) applied to this prover: the first pass derived both challenges from the transcript alone
(`H(root)`, `H(root, w, h, q)`); the public words entered only through the verifier's own `beta` / chain `rhs`, so a
prover could pick the claimed words *after* seeing the challenges. Concretely (`redteam_binding.py`, run on `vy-g3`,
`fixtures/redteam/ligero-binding/v1/binding.json`): with LLL (SymPy) on the 21-VU chain a `delta` with `A delta = 0 mod p` was found in
minutes -- an honest proof of 21 VUs verified against `y16 + delta` (every forged word in `[0, 2^16)`), i.e. any
honest proof proved many false statements. Fix: `statement_digest` -- field, hash, a digest of the compiled
constraint system, `l, n, D, t` (and now `zk, k`), the mode, the chain layout, and every public word (`a, b, c, y`
or `a, b, y16`) -- computed by the **verifier from its configuration** and absorbed into both challenges; sizes
(`l, n, m, t`, path lengths, `q`/`v` shapes) come from the verifier's `Config` / `System`, so a proof of another
shape is `malformed` before any challenge is recomputed. The three fixtures (`fixtures/redteam/ligero-binding/v1/`):
zero-witness proof claiming the real words -- rejected; 2-VU proof against a 4-VU verifier -- rejected (`malformed
proof`); adaptive claim -- accepted by the pre-fix verifier, rejected by the fixed one (`column challenge
mismatch`). The 83-case BabyBear battery (`redteam_battery.py`, `results/babybear_battery.json`): 71 applicable
step/VU cases, **0 forgeries** (12 gadget-level cases do not apply to a unit-level constraint system; the 4
`zero_swap` cases the reference checker rejects are true statements here -- `c = -0` and `c = +0` give the same
output word -- their rejection upstream comes from a stale hint, not a constraint).

**2. The decision workload: 4096 VUs of K = 1536** (run `r20260922-074449-a913`, `results/result_vu4096_b*.json`;
non-ZK, bound statement). 6.6 GB at 85 VUs does not fit 4096 in 48 GB, so the batch is proved as `N_subbatches`
independently committed proofs; `config_for(..., n_proofs)` gives each proof `accounting.per_proof_target` and the
reported soundness is `accounting.union_over_proofs`. Prover time = the sum over the sub-batches, every one verified:

| sub-batching | `t` | per-proof / union (log2) | prover total | **ms/VU** | verifier | transcript | peak device | overhead vs 9.85 ps |
|---|---|---|---|---|---|---|---|---|
| 49 x 85 VUs (l = 8192, n = 32768) | 198 | -134.2 / -128.6 | 8.13 s | 1.98 | 1.59 s | 180 MB | 6.6 GB | 2.0e8 |
| **25 x 170 VUs (l = 16384, n = 65536)** | 196 | -132.9 / -128.3 | **7.97 s** | **1.95** | 1.00 s | 111 MB | 13.2 GB | **2.0e8** |
| 13 x 341 VUs (l = 32768, n = 131072) | 195 | -132.2 / -128.5 | 8.35 s | 2.04 | 0.81 s | 78 MB | 26.4 GB | 2.1e8 |

Split at 25 x 170: hints 0.12 / witness 0.27 / encode 3.62 / transfer 0.11 / Merkle 0.07 / tests 3.61 / openings
0.12 s. Sub-batching costs ~3 columns of `t` and nothing else; the per-VU time *fell* from the first pass's 2.30
ms (the operand marshaling is no longer timed, and the warm loop amortises the graph replays). `l = 32768` is
slower: the size-131072 NTT leaves the L2 sweet spot.

*Route (b), the transposed layout, designed, not implemented*: units as Ligero **rows** (one unit's 3516 witness
values as a row's payload, `l = 3516 -> 4096`, `m = 4096 x 96 = 393k rows` per 4096 VUs -- the 310k x 4096 shape
b-commit measured at 46.7 ms encode+Blake3 and 149 ms linear test per 4096 VUs, 259 ms total). Every constraint
then relates *positions within a row* (`W[unit][var_i] * W[unit][var_j] = W[unit][var_k]`), i.e. the Hadamard and
linear tests become tests over a fixed per-row pattern: the linear test is `sum_i alpha_i p(zeta_i) = beta` per row
(the structure b-lintest exploits -- one interpolation of the pattern, then a batched inner product), the quadratic
test becomes a *permuted* Hadamard product (`p(zeta_a) p(zeta_b) - p(zeta_c)` across positions of the *same* row),
which Ligero does not test directly: it needs either a rotation argument per (a, b, c) triple pattern (3387
patterns) or a re-encoding of the permuted rows -- the cost the row layout avoids. The cross-unit chain becomes a
constraint between *rows* (`p_j(zeta_c) = p_{j+1}(zeta_y)`), a rotation of the row index, which is a cheap
general linear test. Estimate at b-commit's buckets: encode+hash 47 ms and linear test 149 ms per 4096 VUs are measured; the quadratic
test is the open item -- either one re-encoding per permutation class of the 3387 (a, b, c) position patterns, or a
permuted encoding of the whole 393k x 4096 matrix per class (>= 47 ms each). On paper (b) is 0.3-0.5 s per 4096 VUs
against (a)'s 8 s **if** the permuted Hadamard test costs O(1) encodings; until it does, (a) is the measured route
and (b) the next design. The int32 constraint-test rewrite (the 36% term) applies to either.

**3. HVZK** (run `r20260922-081245-1e0d`, `results/result_vu4096_zk_b*.json`, `--zk`; PROTOCOL.md 8b as
implemented; the first `COMPLETE_HVZK_BACKEND` of the campaign):

| | 25 x 170 VUs (l = 16384, k = 16640, n = 65536) | 49 x 85 VUs (l = 8192, k = 8448, n = 32768) |
|---|---|---|
| `t` (union over the sub-batches, log2) | 197 (-128.05) | 200 (-128.20) |
| prover total / **ms/VU** | 8.77 s / **2.14** (+10% over non-ZK 1.95) | 9.23 s / 2.25 |
| of which ZK masks (host SHAKE-256, ~1.5 M symbols per proof) | 0.45 s | 0.73 s |
| encode / tests | 3.84 / 3.74 s | 3.58 / 3.88 s |
| verifier / transcript / peak device | 1.01 s / 122 MB / 13.7 GB | 1.55 s / 193 MB / 6.9 GB |
| overhead vs native 9.85 ps/VU | **2.2e8** | 2.3e8 |

Gate under `--zk` (`results/gate_vu_zk.json`): 16 VUs accept, 52/52 negatives and 252/252 mutations rejected.
Simulator-style test (`redteam_zk.py`, `results/redteam_zk.json`, PASS on the L40S): opened columns uniform (chi-square
64 bins, p = 0.49); a proof of real VUs vs a proof of the trivial VU at the same layout indistinguishable on the
opened columns and on `w, v, h, q` (p = 0.12-0.43); 0 opened values equal to a payload value of their row and 0
constant rows in the clear -- the unmasked control leaks 1018 of 1018 constant rows at every column and 192 k of
665 k opened values are their row's payload. `n` did **not** double (section 8b's estimate assumed `k` rounded to a
power of two; `k = l + 256` in coefficient form keeps `n = 4 l`), so the ZK cost is +10%, not +33%.
**What separates this from `COMPLETE_ZK_BACKEND`**: honest-verifier only -- the coins are Fiat-Shamir of the
transcript; against a malicious verifier the HM96 commitment to the verifier's coins before `root` (one more round,
depth 3, no arithmetic) is not implemented. Also still placeholders: the hash, the SHAKE expander (and the mask
sampler's 2^-31 per-symbol bias).

**4. The 36% term** (run `r20260922-082245-c96b`, `results/result_vu4096_split*_b16384.json`, `tests_bench.py`). The
quadratic test was three `apply_sparse` passes producing `Q x n` intermediates (A, B, C: 3387 x 65536 int64 = 1.8 GB
each), their product, and a D-fold combine -- for a constraint set that is 94% booleanity (`W_j^2 = W_j`: 2534 bit
rows + 610 selectors + 52 others = 3196 of 3387). Split form (`witness.quad_p0`): the booleanity constraints as one
fused `sum_j rho_j (U_j^2 - U_j)` pass over their rows (no intermediates), the 191 general constraints through the
sparse tables of their subset; bit-equal to the naive form (asserted by `tests_bench`, and every gate re-run).
**122 -> 29 ms per proof** at `n = 65536` (the booleanity pass reads 1.68 GB six times in 13 ms); on the 4096-VU
workload (25 x 170, same parameters as above):

| | non-ZK (`t = 196`, union -128.25) | HVZK (`t = 197`, union -128.05) |
|---|---|---|
| prover total / **ms/VU** | 5.63 s / **1.37** (was 1.95) | 6.43 s / **1.57** (was 2.14) |
| split | hints 0.12, witness 0.27, encode 3.62, Merkle 0.07, tests 1.27, openings 0.12 | + zk masks 0.45; encode 3.84, tests 1.41 |
| peak device | 7.9 GB (was 13.2) | 8.5 GB (was 13.7) |
| overhead vs native 9.85 ps/VU | **1.4e8** | **1.6e8** |

The ZK cost is now +14% (0.80 s: 0.45 s host SHAKE for ~1.5 M mask symbols per proof -- a device sampler would
remove most of it; +0.2 s encode for `k = l + 256` and the 4D mask rows; +0.13 s tests). What is left: **encode is
60%** (the int64 radix-2 NTT, 3.6-3.8 s of 5.6-6.4 s), tests 22%, everything else < 10%. The NTT is the next term
(tensor-core DFT per section 8, or int32 Montgomery butterflies in Triton).

Runs: `python -m backends.direct.ligero.run bench-vu [--zk] --batch 16384 --total-vus 4096 --reps 3 ...`,
`gate-vu [--zk]`, `python -m backends.direct.ligero.redteam_zk`, `redteam_binding`, `redteam_battery`, `tests_bench`.

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

## Files

`compile.py` (Expr, LigeroCtx, the integer audit, lookups as one-hot selectors) . `unit.py` (the unit and chain
systems; public vectors) . `field.py` (BabyBear, NTT, coset encoding) . `merkle.py` (GPU Blake3 via `hash_gpu`;
host Blake3/SHA-256 fallback) . `hints_torch.py` (vectorised `gen_step`) .
`witness.py` (W from the program; sparse constraint tables) . `chain.py` (layout, the chain test's coefficients) .
`protocol.py` (prove / verify) . `vu.py` (chain gate / bench) . `run.py` (CLI) . `encoder_bench.py` .
`results/` (gate logs, result envelopes). Ledger: `backends/numerical/reports/ledger/b-ligero.jsonl`, plot
`backends/numerical/reports/plots/overhead_b-ligero.png`.
