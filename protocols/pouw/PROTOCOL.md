# Proof of useful work

Certifies that a declared workload of integer or FP8 matmuls was computed with at least a stated fraction of the honest
work, by sampled recomputation of committed checked values. The setting and the cost model (W1 on an RTX 4090) are the
PoUW problem statement's; the guarantee is Lean's `Pouw.Proofs.endToEnd` instantiated with a scheme's conjecture.

## The interface

A scheme (`protocol.PoUWScheme`) is a named, versioned object, looked up by name (`schemes.get`), with pinned vectors under
`tests/vectors`, like `verity.commitments.scheme.CommitmentScheme`. It turns one declared matmul's committed operand
rows into, for each tile, checked values, a transcript leaf and the useful output. It exposes:

- `name` and `certificate`: what an accepted audit rests on (the conjecture's claim id, γ_0, Ω*, the pinned Lean theorems,
  the hypotheses, the allowances);
- `check(matmul)`: refuses a declaration outside the domain, before any work;
- `tiles(shape)`: the checked units, a partition of the output cells; a tile is rows of A times rows of B at full k;
- `wref(shape)`: the honest reference's W1 cost, from the layout only;
- `noise(ctx)`: the matmul's hash-derived values, from the salt, the declaration and the committed roots;
- `encode(ctx, noise, a_rows, b_rows, tile)`: the checked product's operands for a tile, from its strips, or nothing when a
  row is malformed;
- `admissible(ctx, ops)`: the input policy (worst-case-input schemes accept every well-formed tile);
- `checked(ctx, ops)`, `digest(ctx, noise, checked)`, `useful(ctx, noise, ops, checked)`.

Operands are committed as rows along k: A is m × k and the weight B is committed as its n rows, so the product is A·Bᵀ.
Everything else is the protocol's (`audit`), the same for every scheme.

## Lifecycle (`audit`)

1. **Weights.** Each weight's rows are committed in a `verity.commitments.merkle` tree bound to the weight id (owner −2);
   `weights_root` binds every weight root.
2. **Salt.** `derive(beacon, "verity/pouw/salt/v1", {"weights": weights_root})`, drawn after the weights, 32 bytes.
3. **Units.** For each matmul (a 64-bit index, a shape, a weight id), the activation rows are committed bound to the salt
   and the index. The scheme derives its noise from (salt, matmul, roots); every tile yields checked values and a digest.
   The useful output is assembled from the tiles.
4. **Transcript.** A tree over every tile's digest, in matmul order then tile order, bound to (salt, weights_root).
5. **Selection.**
   - `Sampled(t)`, the default: `derive(source, "verity/pouw/audit/v1", {"transcript": root, "salt": salt})`; draw j is
     `key.uniform(W, j)` mapped to a tile by cumulative work, so t tiles are drawn independently, each with probability
     W_ref(tile)/W, a tile's work being its cells' share of its matmul's W_ref (Lean `WorkWeightedSampling`).
   - `Lottery(target)`, Pearl's mining rule: the tiles whose digest, read little-endian, is at most target·|rows|·|cols|·k.
6. **Check.** For each selected tile the prover opens A's rows, B's rows and the leaf. The verifier checks the openings,
   recomputes the tile through the scheme (`encode`, `admissible`, `checked`, `digest`) and compares the leaf.
7. **Profile.** When every selected tile passes, `WorkProfile`: for every ε_s in (0, 1] and every operator whose online cost
   is at most T, `Pr[accept ∧ T < (1 − γ)(1 − ε_s)·W] ≤ (1 − ε_s)^t + η` (Lean `endToEnd`), with γ = 1 − (1 − γ_0)/Ω*,
   under the certificate's conjecture. A scheme with no γ_0 gets only the sampling half; a lottery gets neither.

**Randomness.** The protocol owns the outer coins: the salt (from a beacon round) and the draw (from the verifier's source,
never the prover's root). A scheme's noise is its own derivation from the salt and the prover's roots, Fiat-Shamir inside
the scheme, whose grinding is inside the conjecture's error term.

## `ncp-v1`: noise-cancelling polarization under TTNCP_U(0.5%), route U at β = 128

For int7 A and B (entries in [−64, 63], one two's-complement byte each):

~~~text
E₁ ∈ [−31, 32]^(m×k)   per unit; row i = low 6 bits − 31 of SHAKE256("verity/pouw/ncp/E1/v1" ‖ 0 ‖ frame(salt, index, root_A, weight, i))
F₁ ∈ [−31, 32]^(k×n)   per (epoch, weight); column j from SHAKE256("verity/pouw/ncp/F1/v1" ‖ 0 ‖ frame(salt, weight, j))
R                      signed permutation with R[π(j)][j] = σ_j:  E₂[i][j] = σ_j·E₁[i][π(j)],  F₂[π(l)][j] = −σ_l·F₁[l][j]
E₃ = E₁ + E₂,  F₃ = F₁ + F₂
X = [A + E₁ | A + E₂ | −(A + E₃)],   Y = [B + F₁ ; B + F₂ ; B + F₃],   X·Y = A·Bᵀ
β        128: X + β is u8 and Y is s8 on the int7 domain (Lean `rangeU`, `biasU8Iff`)
c₀[j]    −β·Σ_{l<3k} Y[l][j]   (−3β·colsum(B) for an unsigned P), the accumulator's initial value
checked  C^U_t = c₀ + Σ_{l<16t} (X + β)[:, l]·Y[l, :] mod 2^32, t = 1 … 3k/16 (Lean `mmWU 128`); the last is A·Bᵀ
digest   SHAKE256("verity/pouw/ncp/tile/v1" ‖ 0 ‖ words)[:32], words little-endian, step-major, row-major within a step
tiles    16 × 16
domain   16 | m;  64 | k, n;  1,088 ≤ k ≤ 2^16  (γ_0 = 1/200 ≥ 16/(3k));  n₀ = 1,600 ≤ n ≤ 2^16  (Ω* = 1 + f/(3n₀) ≤ 1.005)
wref     3·m·k·n + f·m·k, f = 24   (route U under W1 with sub-word moves priced, Lean `wrefU 24`; the per-weight forming of
         B + F_i is left out)
γ        1 − (199/200)/(1 + 8/n₀) ≈ 0.995% over the whole domain (`gammaFromTTNCP_U_v1`); ≈ 0.89% at n = 2,048; ≈ 0.69% at
         4,096²; the same formula gives 1.27% at n = 1,024, so the certificate records n₀ with β and f
~~~

- **Checked values.** Route U's accumulator, the u8 × s8 tensor-core schedule the W1 accounting prices. It equals the
  unbiased running sums shifted by public data, C^U_t = C_t − β·Σ_{l ≥ 16t} Y[l][j] (`checkedU_eq`), so every bound
  proved modulo V₀ transfers. The unbiased sums (`checked_unbiased`) are retired as checked values; their honest program
  would cost 36 units per position (γ = 1.08% at n = 2,048).
- **β = 129** (f = 20, γ ≈ 0.91% at n₀ = 1,600, 0.82% at n = 2,048; `gammaFromTTNCP_U_C2`) needs an unsigned P and a
  SASS check that the IADD3 with two negated sources is one instruction. The native route-U kernel
  (`kernels/pouw_gemm.cu`, `ncp_route_u`) reads X + β already formed, so its SASS shows the u8 × s8 `IMMA` but not the
  forming step; `ncp-v1` stays at β = 128.
- **γ is conditional on TTNCP_U(0.5%).** The red team found that A2, as first written, required the model program to
  reproduce every output, not only the checked words. Restricted to the checked words, A2 makes the Theorem D chain
  equivalent to TT_NCP rather than a reduction to milder assumptions (`a2WordsIff`). So TTNCP_U, TT_NCP over the route-U
  checked values, stays the named conjecture, and every γ here holds only under it.

- **P.** `ncp-v1` uses the word-granular production P v1.1: a SHAKE256 Fisher-Yates over the k/4 aligned words from
  seed `pous-pouw-ncp-Pword-v1`; repair 1 (seed + `/repair`) so that no word lands in its own 16-step; repair 2 (seed +
  `/repair2`): while some word lies on a cycle of the arcs w → p(p(w)) that stay inside w's step, swap p at the least such
  word with p at a drawn word, under repair 1's condition. The order inside each word is kept and every sign is +1. Its
  SHA-256 (entries as 2-byte little-endian integers) is `536757a2…` at k = 1,088, `7f00a9b7…` at 2,048, `24e68f30…` at
  4,096 and `a57ea054…` at 11,008, all reproduced by the red team's own generator, which certified independence rank
  T − 1, λ₃ = 32 and MinRank 16 at each. The repairs guarantee clause 1 of the rejection rule and minrank.md's hypotheses
  (a) to (c), not the rule's other clauses or λ₃ ≥ 16: the rule (Lean `Admissible`) is checked at every k, and
  `tests/test_pouw_word_p.py` certifies all of them exactly at the k the MVP uses (1,088, 2,048, 11,008).
  `ncp-v1-shift24` uses the shift by 24 instead, admissible with independent checkpoints for every k ≥ 1,024 with
  16 ∣ k (Lean `shiftBy24_admissible`, `independentWitness`).
- **Lean** (the certificate's pins, all approved by the red team, Phase 13):
  - the reduction: `Pouw.NCP.RouteUProofs.gammaFromTTNCP_U_v1` (β = 128, f = 24, n₀ ≤ n), from the named assumption
    `Pouw.NCP.Assumptions.TTNCP_U`, with `ttncpUIff` (it is TT_NCP over the route-U checked set) and `ttncpUWitness`;
  - the algebra: `Pouw.NCP.Proofs.cancel` and `stacked` (the unbiased stack, which the final checkpoint equals), and
    `Pouw.NCP.RouteUProofs.checkedU_eq`, `checkedU_final`, `wrapU`, `rangeU`, `mmWUExact`, `yuChecked`;
  - the chain to the checked words: `Pouw.NCP.RouteUWordProofs.int7Int32UPos` (k ≤ 2^16 for an unsigned P, 2^15 for a
    signed one), `ncpWordFloorIndU`, `wrapFinalOnlyIndU`, `wrapWordFloorIndU`;
  - the barrier and the generic game: `Pouw.Barrier.Proofs.generic`, `theorem1`, `workWeightedSampling`, `endToEnd`.

  Assumed: TTNCP_U(0.5%); `Word.NzNCP` of the route-U words (the floors' hypothesis). Not proved: that M_4090 runs H_ref
  within `wref` (`AdmitsRef`), and the byte-level noise expansion here (the Lean takes any expansion of H's answers with
  the stated ranges).

## `pearl-fp8-v4`: Pearl's FP8 scheme (`schemes/pearl_kw.py`)

Pearl's certificate-v4 FP8 scheme, as published: the whitepaper "Pearl Floating Point Scheme Specification" (Sep 2026)
and pearl-research-labs/pearl PR #311 at `284b147b`, whose verifier code rules where the two differ. The module's
docstring states every parameter.

- **Conformance.** `tests/vectors/pearl_kw.json` holds Pearl's own outputs, computed by its Rust at `284b147b` from
  inputs `tests/pearl_vectors.py` derives; the generator is `tests/pearl_vectors.rs`. For both of Pearl's devices, H100
  and B200, the reference reproduces:
  - the labelled hashes, the noise lines and the BF16 operations;
  - the device matmul and its partials;
  - the census (λ, the grid, the skip);
  - the lane layout;
  - on whole 16 × 16 tiles at k = 1,024: the FP10 norms, the noise, the quantized codes and the per-row scales, the
    accumulator, the extract, J, and the jackpot policy's verdict (one tile approved, one rejected for dead entries).
- **The RTX 4090.** Pearl's whitelist has no Ada device, so `ada` is ours: Verity's pinned `sm89.mma.m16n8k32.e4m3`
  atom (`verity.ml.tc`), with H100's structure (a fresh accumulator every 128 products, promoted by one FP32 add), δ = 1
  and W = 13. With plain chained `mma.sync` instead, the 14-bit adder anchored at the running sum fails Pearl's
  unpredictable-summands census on honest inputs.
- **What the protocol replaces.** The chain headers become the salt. The activation and weight roots stand in for
  Pearl's keyed-BLAKE3 Merkle openings. The seed order is kept.
- **The conjecture.** Pearl's Assumption 1 with the jackpot policy. It names no γ_0, so the profile carries only the
  sampling half and the policy's allowances (5% skippable summands, 1/64 dead entries per side, the unchecked peel).
- **Its differences from milestone 1's D1**, the hardened KW transcript: see the module docstring.

- **Vectors.** `ncp-v1`'s checked values equal, bit for bit, Lean's own `#eval` of
  `mmWU 128 k A B E₁ F₁ R (blockEnd 16 (3k) t)` at every checkpoint, and the retired unbiased form equals Lean's
  `mmW (blockEnd 16 (3k) t) (stackX k A E₁ R) (stackY k B R F₁)`, for the word-granular P, the shift by 24 and a random
  signed permutation (`tests/vectors/ncp_lean.json`, from `tests/ncp_lean.py` on the integrated package, with the Lean
  sources' SHA-256). The scheme's own pinned vectors are `tests/vectors/ncp.json`.

## `fp8-is-h100-*`: fresh FP8 atoms and FP32 running words on the H100 (`schemes/fp8_is.py`)

Genuine FP8 PoUW: the checked computation is the H100's E4M3 tensor-core matmul with FP32 accumulation. The scheme is
a fixed **core** over a swappable **forming layer**; `fp8-is-h100-<layer>` names the pair. **No certificate names a
γ_0 yet**: `DistinctLive` is refuted for `fpncp-v0` and `d2-draft`, and for `d3-v0` it is claimed but not yet
certified (Phase 17c).

~~~text
rows      k E4M3 codes, one byte each; 0x7F and 0xFF (NaN) are malformed; the layer may narrow the domain (`admits`)
forming   the layer gives each tile's stacked operands X̃ (rows) and Ỹ (columns), K = blocks·k codes, block-major,
          and its own checked words for those rows and columns (`Forming.form`)
atom      I_τ = HOPPER_E4M3_WGMMA_K32.step(+0, X̃[i, S_τ], Ỹ[S_τ, j]), S_τ the τ-th 32-slice of K, ascending
          (wgmma m64nNk32 f32.e4m3.e4m3, pinned on H100 with nonzero accumulators too)
running   R_τ = add.rn.f32(R_(τ−1), I_τ)   (`verity.ml.tc.fp32.add`: binary32, ties to even, no FTZ)
          R_0 = +0 for an unseeded layer, and then R_1 = I_1 is not checked (a duplicate in every output,
          `c2FirstRunCopy`); a seeded layer gives each cell a salt word R_0, and R_1 is checked
checked   the layer's words, then every I_τ and every checked R_τ; useful = R_T (less R_0 when seeded, as glue)
digest    SHAKE256("verity/pouw/<name>/tile" ‖ 0 ‖ layer words of the tile's rows, then columns ‖ core words)[:32];
          core words little-endian, step-major, row-major over cells within a step, I_τ before R_τ
tiles     64 × 64 (one m64n64k32 atom per slice: Round 7 measured 32 per word at N ≥ 64, and m64n8 at 22% of peak)
domain    64 | m, 64 | n, 32 | k, 8,192 ≤ m, k, n ≤ 2^16
wref      ⌈m·n·(32·T + p_FADD·n_R) + f_A·m·k + f_B·n·k⌉: an atom word per cell per slice at 32, an FADD per checked R_τ at
          the price table's rate (n_R = T − 1 unseeded, T seeded), and the layer's trace
Ω*        (32T + p_FADD·n_R + f_A + f_B)/(32T + 32·n_R + c_A + c_B) at m = n = k = N (the scheme's `mnk_min`), c being
          the part of f spent on the layer's checked words
γ         1 − (1 − γ_core)/Ω*,  γ_core ≤ ε + μ·(32 − c_k8)/32  (ε from `DistinctLive(ε)`, μ the fraction of credited
          words a sub-32 writer can write, c_k8 its measured price; `distinctWritesMH100`, `distinctWritesLegacy`)
~~~

- **The core's premise** (the lower-bound lane's cost model, `genuine-fp8-interface.md` §1.3): under `H_32` every
  checked word that is salt-dependent and distinct from the others costs at least 32 (`distinctWritesMH100`, proved;
  `distinctWritesAttained`), and the honest schedule pays exactly 32. What is left is a property of the forming layer,
  `DistinctLive(γ_0)`: for every admitted input, all but a γ_0 fraction of the credited words are salt-dependent and
  pairwise distinct as functions of the salt. Plus `A2_fp` (branching, routing, grinding) and `AdmitsRef`.
- **`H_32`** is stated for every writer class, including legacy `mma.sync` small-K shapes, sparse `wgmma`, the b1 paths,
  conversions, the uniform datapath and memory-side arithmetic (`red`/`atom`, `cp.reduce.async.bulk`, texture
  filtering), each priced per 32-bit word it writes; it is a measurement on an H100 SXM, not yet made.
- **`d3m-v0`** and **`d3m-v0-decode`** (`D3m`), **the lead:** D-3 with Track B's max law, on the committed X_q.
  - **The noise.** C-2's three NCP blocks formed directly on X_q's codes. Every entry's noise amplitude is
    max(σ, one E4M3 cell of the entry, one cell of its NCP partner), with σ = max(rms/16, 28/√k), from one f16 noise
    field. The floor 28/√k gives all-zero and non-normalized rows and columns noise that survives rounding (X-D3m-4).
    The cell is read on the cast's own grid (X-17d-3).
  - **The arithmetic.** Block values are HADD2.F32 and the casts are `cvt.rn.satfinite.e4m3x2.f32`, so a block value at
    or beyond ±464 saturates to ±448.
  - **The streams.** Weight streams are keyed per unit and per column, so B̃ is re-formed per unit: r = 1, the only reuse
    D-3m tolerates (X-D3-3).
  - **The price** is schedule M4 (the D-3m strike) from `PRICES`: 426.27 per activation element at Round 8. The weight
    side costs 290 with a registered FP16 weight copy (`d3m-v0`) or 322 decoded every epoch (`d3m-v0-decode`); that is
    Daniel's pending ruling (ii). The two share one checked function and one digest tag.
  - **At 16,384³ with r = 1:** Ω* gives γ = 0.972% and 1.004% at γ₀ = 1/400.
- **`d3-v0`** (`D3`), superseded by D-3m: repaired C-2, track D's D-3. It keeps C-2's three NCP blocks and row-RMS
  noise, and adds partner-max full-cell noise, s_l·v_l with s_l the larger E4M3 cell of the entry and of the entry that
  reads the same noise in block 2, which fixes X-C2-B1. Weight streams are keyed per column (X-D2-5).
  - It is formed in binary16 (`verity.ml.tc.fp16`, the half2 pipe), with a power-of-two row scale α and E4M3 casts by
    `cvt.rn.satfinite.e4m3x2.f16x2`. The core runs unseeded.
  - Its op-by-op MH100 trace (`D3_TRACE`) is 372 units per element per side nominal and 494.65 at Round 7's measured
    prices. Track D claims 192 for a native schedule; `D3(form_units=…)` sets the price the certificate uses.
- **`fpncp-v0`**: C-2's FP-NCP forming as posted (row-scaled uniform noise, NCP blocks, three `cvt.rn.satfinite.e4m3x2`),
  with every FP32 operation fixed (`FpNcpV0`). **Refuted:** its noise is row-scaled, so it is rounded away on large
  entries (X-C2-B1) and alignment erases everything but a spike (X-D1-1). Kept as the negative control; the tests
  reproduce both attacks bit for bit. Its MH100 trace is 384 units per element per side.
- **`d2-draft`**: track D's D-2 (`D2Draft`); **fell in Phase 17b** (X-D2-1 … 8), kept as the worked example of a layer
  with checked forming words. Each NCP block's operand is a checked FP8 Hadamard transform (block 32,
  α = 1/8 left, β = 1/4 right, αβ·32 = 1), computed as chains of k32 atoms that mix the base and E4M3 noise codes in
  aligned 4-code words, then converted by `cvt.rn.satfinite.e4m3`; block 3 is the three-part transform, and signs sit in
  the transform weights. Its checked words are every transform atom word and every converted code (7 words and 3 codes
  per element: 272 units credited per element per side). The noise-code distribution (`MaxRelativeCodes`, 64 units) is
  a **placeholder** for track D's repair. The domain is |A| ≤ 64 and |B| ≤ 32, so no transform saturates.
- **Prices.** `PRICES` holds Round 7 as measured and Round 8 loop-free (FADD 32, half2 36, INT 64, F2FP 64 per
  instruction). The X_q unpack and HADD2.F32 are the red team's assumptions until Round 9 measures them. The core's FADD
  and D3m's trace read the same table (`prices=`, default Round 8).
- **Vectors.** `tests/vectors/fp8_is.json` (from `tests/fp8_is_vectors.py`): every layer on a test instance, and `d3-v0`
  and `d2-draft` at k = 8,192. The core is checked against the total FP8 step and numpy's float32 running sum, and D-2's
  exact algebra (before any rounding) against A·Bᵀ.

## Not here yet

- The fast implementations: the RTX 4090 kernels and the vLLM adapter (`integrations/vllm/verity_vllm/protocol_options/pouw.py`)
  are gated by these vectors when they land.
- The per-weight share of the honest reference's work in `wref`, and a `TargetProfile` Definition of the NCP GEMM.
