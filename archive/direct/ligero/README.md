# b-ligero: a specialised Ligero-style direct-constraint prover for the transition unit (torch, GPU)

Track B, lanes `b-ligero` (first pass, sections below) and `b-ligero2` (second pass, next section: bound statement,
the 4096-VU decision workload, HVZK). Design: `PROTOCOL.md`. Proof class **NON_ZK_PROOF_DIAGNOSTIC** without `--zk`,
**COMPLETE_HVZK_BACKEND** with `--zk --no-coin-commitment`, **COMPLETE_ZK_BACKEND** with `--zk` (lane b-zk, third
pass below: HM96 coin commitment, PROTOCOL.md 8c; hash and challenge expander are host placeholders; the relation is
WP2's checker, nothing dropped). Every number below was measured on `vy-g3` (NVIDIA L40S 48 GB, torch cu124) on the frozen
input set `fixtures/bench-instances/v1`.

## Third pass (lane b-zk, ledger `b-zk.jsonl`): COMPLETE_ZK_BACKEND, 4.6x

Runs `r20260922-090407-5483` (SIMT encoder), `r20260922-091350-15e3` (+ device mask sampler),
`r20260922-092404-2a1b` (+ fused test kernels; the numbers below). 4096 VUs of K = 1536 as 25 sub-batches of 170
(`l = 16384`, `n = 65536`, `D = 6`), median of 3 reps, verifier accepted every proof of every rep; every gate green
before and after each step (52/52 negatives, 252/252 chain mutations, 243/243 unit mutations, with and without
`--zk`; `redteam_zk` PASS incl. the malicious-verifier section; `redteam_binding` OK).

| 4096 VUs, L40S | non-ZK | HVZK (`--zk --no-coin-commitment`) | **ZK** (`--zk`, PROTOCOL 8c) |
|---|---|---|---|
| prover total / **ms/VU** | 1.37 s / **0.334** | 1.43 s / **0.350** | **1.41 s / 0.345** |
| overhead vs native 9.85 ps | 3.4e7 | 3.6e7 | **3.5e7** |
| ZK cost over non-ZK | -- | +4% | **+3%** |
| verifier | 0.96 s | 0.94 s | 0.96 s |
| transcript | 110.8 MB | 122.3 MB | 122.3 MB |
| `rounds.sequential_depth` | 3 (commitment on in every mode) | 2 | **3** |
| peak device memory | 2.81 GB | 3.53 GB | 3.53 GB |
| t / union bound (`accounting.union_over_proofs`, 25 proofs) | 196 / 2^-128.25 | 197 / 2^-128.05 | 197 / **2^-128.05** (incl. the 2^-190 coin-hiding term) |
| proof class | NON_ZK_PROOF_DIAGNOSTIC | COMPLETE_HVZK_BACKEND | **COMPLETE_ZK_BACKEND** |

Split of the ZK prover (s over the 25 proofs): hints 0.12, witness 0.26, masks 0.02, encode 0.20, Merkle 0.06,
tests 0.57 (`tests_w` 0.34 = challenge-1 host expansion + the D-fold combinations; `tests_quad` 0.13; `tests_chain`
0.10), openings 0.12. Was (b-ligero2 HVZK): 6.43 s = encode 3.84 + tests 3.74 (later 1.41) + masks 0.45 + ...

**1. HM96 coin commitment (PROTOCOL.md 8c; `protocol.Coins`, `coin_commit`, `CoinOpeningError`).** Step 0 of the
interactive protocol: the verifier commits to the randomness of BOTH of its coins, `c_i = BLAKE2b(COIN_TAG || r_i || s_i)` (32-byte `r_i`,
32-byte salt `s_i`), before the prover's first message; `(c_1, c_2)` are absorbed into every
challenge; slot 1 is opened after `root` (the prover checks `c_1` and aborts otherwise), slot 2 after `w, h, q, v`.
Challenge `i` is `H(stmt, c_1, c_2, transcript so far, r_i)`. In the Fiat-Shamir transcript the prover plays step 0
(`Coins.sample`) and the proof carries `coin_commit` and `coins`; the verifier recomputes the commitments and rejects
a foreign or mismatched opening. What the malicious verifier loses: it can no longer choose `r`, `rho`, the columns
as a function of `root` (or of `w, h, q, v`); the coins are fixed (information-theoretically hidden, computationally
bound) before it sees anything, so its view equals the honest-verifier view at those coins, which 8b's masking
simulates. The simulator runs the verifier to its commitment, rewinds once to learn the opening (extractable because
the verifier must open to continue), then produces the 8b honest-verifier transcript for those coins.
`rounds.sequential_depth` 2 -> 3 (`c`, `r_1`, `r_2`). Cost: 1.41 vs 1.43 s -- nothing measurable.

Open questions of `zk_cost.md`, for this prover: *pole redraw* -- B-Ligero has no retry-on-bad-challenge path (the
quotient `h = P_0 / Z_H` is over a fixed domain with no poles; the column sampler draws `t` distinct indices by
rejection from the hash stream, which depends on public data only), so nothing is redrawn under committed coins.
*Abort leakage* -- the prover's only aborts are the two `CoinOpeningError`s (public data: the verifier's opening vs
its commitment) and, with `check=True`, the self-check assertions on the honest witness (a real deployment runs
`check=False`; the assertions can only fire on an invalid witness, which is not a ZK input). No abort depends on the
witness. `redteam_zk.py` section 4: an equivocating verifier is aborted at slot 1 / slot 2 with only `root` /
`root, w, h, q, v` sent (never opened columns); adversarially chosen but committed coins (all-zero, or targeting a
fixed column set) give chi-squared p > 0.001 on every message between a real and an all-trivial witness and the
opened columns stay uniform; the pre-8c proof is rejected by the 8c verifier (`column challenge mismatch`).
Accounting: `protocol.soundness` adds `COIN_HIDING_TERM = 2 * 2^64 / 2^256` (two slots, brute-force of a 32-byte
`r`) -- the union bound moves by < 2^-60.

**2. Device mask sampler (`mask_sampler.py`).** ChaCha20 in counter mode (RFC 8439 block function as a cupy kernel;
the host reference reproduces the RFC test vector) with exact rejection: a 32-bit word is accepted iff `u < 2p`,
then reduced mod p (acceptance 0.9375, exactly uniform -- the 2^-31 relative bias of `u mod p` is gone, so the 8b
argument needs no bias term). The host fallback uses the same rejection on SHAKE-256. This is a documented new
derivation (not bit-exact with the old `SHAKE mod p` sampler; the masks are secret randomness, nothing depends on
their derivation). 0.44 -> 0.02 s.

**3. SIMT encoder (`encode_simt.py`, `encode_simt_test.py`).** The prover is over BabyBear, so the b-encode kernel
applies: the int32 Montgomery radix-4 DIF/DIT from `encode_ntt_simt` (v2), adapted to Ligero's domain (inverse NTT
on H then the four cosets `g w_4^c <w_l>`, output in the natural order of `g <w_n>`), with the ZK mask `Z_H s_j`
folded in per coset (`z_c s_{j,i} (g w_n^c)^i` added to the bit-reversed coefficients) and an optional natural-order
coefficient output (for `interp_H(W_j)`); dynamic shared memory for `l = 16384`. Bit-exact against `field.encode` /
`eval_on_coset` / `intt` (plain and ZK, l = 256..16384). Encode bucket, 3516 rows x l -> 4l on the L40S: torch int64
**145.0 ms -> SIMT 9.60 ms** at `l = 16384` (15x; 96 GB/s of codeword, shared-memory bound at K = 65536), 68.4 ->
2.05 ms at `l = 8192` (33x; 225 GB/s). On the 4096-VU workload: 3.62 s -> 0.20 s. L40S vs 4090: the 4090 number
(`b_encode_4090.md`, 32.2 ms per 310,602 rows of the smaller v2 domain) is not the same shape; per codeword byte the
L40S at `l = 8192` runs at 225 GB/s against the 4090's ~0.78 x 1008 GB/s = 790 GB/s, i.e. this kernel on the L40S is
~3.5x below the 4090's HBM-floor fraction -- it is shared-memory / issue bound at these row lengths, not bandwidth
bound (the 4090 measurement was at n = 1024-4096 per row).

**4. Fused test kernels (`tests_fused.py`, `tests_fused_test.py`; step 5 of the brief).** After the encoder the
tests were 52% (1.37 s). Two cupy kernels, one thread per column, rows split across blocks (>= 2^19 threads in
flight), 64-bit accumulation of three products per reduction: `lincomb` (all D combinations of a row matrix in one
pass, also the `x^2 - x` booleanity form) and `quad_general` (the non-booleanity quadratic constraints straight from
CSR tables, A_q B_q - C_q and the D-fold rho combination without the (Q, n) intermediates). Bit-exact vs the torch
paths; 3521 x 16640 combine 3.94 -> 0.76 ms, 2534 x 65536 booleanity 14.3 -> 1.05 ms, 853 general constraints on
65536 columns 41.8 -> 2.26 ms. Tests 1.37 -> 0.57 s; prover 2.22 -> 1.41 s.

**Non-ZK note.** The coin commitment is on in every mode (it costs nothing), so the non-ZK row also reports depth 3;
it is not needed there.

## Second pass (lane b-ligero2, ledger `b-ligero2.jsonl`)

**1. Statement binding (SOUNDNESS fix, commit 302832f).** The red team's checklist (`note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu` F1/F2/F19,
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
mismatch`). The 83-case BabyBear battery (`redteam_battery.py`, `notes-asset:campaigns/r20-proof/assets/b-ligero/results/babybear_battery.json`): 71 applicable
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

Gate under `--zk` (`notes-asset:campaigns/r20-proof/assets/b-ligero/results/gate_vu_zk.json`): 16 VUs accept, 52/52 negatives and 252/252 mutations rejected.
Simulator-style test (`redteam_zk.py`, `notes-asset:campaigns/r20-proof/assets/b-ligero/results/redteam_zk.json`, PASS on the L40S): opened columns uniform (chi-square
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

Runs: `python -m archive.direct.ligero.run bench-vu [--zk] --batch 16384 --total-vus 4096 --reps 3 ...`,
`gate-vu [--zk]`, `python -m archive.direct.ligero.redteam_zk`, `redteam_binding`, `redteam_battery`, `tests_bench`.

## What it is

* The unit's witness is one column of `W` (m rows x l units): 99 pinned decode rows, 92 hint rows (the checker's
  `gen_step` hints), 170 product rows, **2401 bit rows** (every range check as an explicit decomposition), 610
  one-hot selector rows (the four fixed tables `POW`, `ALIGN`, `LEAD`, `NORM` as one-hot lookups over their
  distinct rows) -- **3372 rows per unit**, 338 linear + 3246 quadratic constraints (the census's 358 mul / 137 lin /
  306 lookups, with the lookups paid as bits and selectors).
* BabyBear, rate-1/4 Reed-Solomon (`[16384, 4096]`) by a batched radix-2 int64 NTT in torch, Blake3 Merkle over
  the codeword columns on the GPU (`verity/ml/kernels/hash_gpu`; host Blake3 fallback), Ligero's three tests (interleaved proximity; linear constraints with the
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
# on vy-g3, /workspace/bl/src, PYTHONPATH=.:benchmarks/numerical/python
python -m archive.direct.ligero.run gate     --root fixtures/bench-instances/v1 --device cuda --vus 64   # notes-asset:campaigns/r20-proof/assets/b-ligero/results/gate_unit.log
python -m archive.direct.ligero.run bench    --root fixtures/bench-instances/v1 --device cuda --reps 5 --out results/result_unit_b4096.json
python -m archive.direct.ligero.run gate-vu  --root fixtures/bench-instances/v1 --device cuda --vus 64   # notes-asset:campaigns/r20-proof/assets/b-ligero/results/gate_vu.json
python -m archive.direct.ligero.run bench-vu --root fixtures/bench-instances/v1 --device cuda --reps 5 --out results/result_vu_42.json
python -m archive.direct.ligero.run bench-vu --root fixtures/bench-instances/v1 --device cuda --reps 5 --batch 8192 --vus 85 --out results/result_vu_85.json
python -m archive.direct.ligero.encoder_bench                                                          # NTT vs INT8 matmul DFT
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

Final state (torch hint generator, GPU Blake3 Merkle from `verity/ml/kernels/hash_gpu`, `torch.compile`d NTT butterfly
and constraint kernels, CUDA-graph replay of the witness program):

| | unit proof, B = 4096 units | VU proof, 42 VUs (l = 4096) | VU proof, 85 VUs (l = 8192) |
|---|---|---|---|
| **prover total** | **0.095 s** (**23.2 us/unit**, 2.2 ms per 96 units) | **0.108 s** (**2.57 ms/VU**) | **0.195 s** (**2.30 ms/VU**) |
| **prover total on the A100** (anchor-a100 lane, NVIDIA A100-SXM4-80GB, run `r20260922-081113-f56e`, same commands, 7 reps, verifier accepted every rep; gate-vu 52/52 negatives + 252/252 mutations rejected) | **0.072 s** (**17.6 us/unit**; 1.32x the L40S) | **0.084 s** (**1.99 ms/VU**; 1.29x) | **0.151 s** (**1.78 ms/VU** = overhead 1.81e8 vs the same card's BF16 peak; 1.29x) -- encode 0.044 s, tests 0.048, hints 0.036, witness 0.016; `note:r20-proof/anchor-a100/20260922T0820Z-report-anchor-a100` section 7 |
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
uv run --frozen research run --on vy-g2 --source . --project verity --stage b_e2e_v2.gate  --require-result -- bash archive/direct/ligero/v2/pod_gate.sh
uv run --frozen research run --on vy-g2 --source . --project verity --stage b_e2e_v2.bench --require-result --env LS="16384 32768" -- bash archive/direct/ligero/v2/pod_bench.sh
# laptop: the CPU toy of the whole protocol
python -m pytest archive/direct/ligero/v2/tests/test_toy.py
~~~

Gate (`r20260922-091001-68f7`): 8 honest `vu-k1536` VUs accepted with the recorded words (every packed intermediate
= the recorded accumulator; prover self-check: every constraint satisfied); `vu-k1536-neg` 52/52 + flipped word
rejected; 39/39 unit + chain mutations rejected; battery `epilogue` 2/2 rejected (`chain` class vacuous under v2).

End to end, 4 096 VUs per batch as N sub-batch proofs (`bench`, all sub-batches proved and verified; l = 131 072 would exceed the 24 GB card, so 4 096 VUs in one proof needs the table-side fix below):

| l (units/proof) | VUs/proof | N | prover s/proof | **ms/VU** | s per 4 096 VUs | overhead | verifier s/proof | proof MB | rows m | peak GiB | union security | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 384 | 170 | 25 | 0.727 | **4.27** | 17.5 | 4.3e8 | 0.26 | 4.6 | 3 139 | 8.5 | 2^-128.3 (t = 196 per proof) | r20260922-094956-5453 |
| 32 768 | 341 | 13 | 0.888 | **2.60** | 10.7 | 2.6e8 | 0.29 | 5.7 | 2 019 | 11.8 | 2^-124.5 (t = 189; t = 193 would meet 2^-128) | r20260922-094315-8be5 |
| **65 536** | 682 | 7 | 1.261 | **1.85** | **7.57** | **1.9e8** | 0.40 | 9.2 | 1 473 | 17.1 | **2^-128.1** (t = 193 per proof) | r20260922-094956-5453 |

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
`notes-asset:campaigns/r20-proof/assets/synthesis/reports/plots/overhead_b-ligero.png`.
