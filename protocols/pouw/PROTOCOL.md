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
leaf      two-level: the tile's 16 × 16 subtiles, row-major (subtile (a, b) = rows 16a … 16a + 15 and columns
          16b … 16b + 15 of the tile, before (a, b + 1); a tile under 16 on a side is one subtile), each digested
          sub(a, b) = SHAKE256("verity/pouw/fp8-is-h100-<tag>/subtile" ‖ 0x00 ‖ layer words of its rows, then columns
          ‖ its core words)[:32], core words little-endian, step-major, row-major over its cells within a step, I_τ
          before R_τ; the Merkle leaf is SHAKE256("verity/pouw/fp8-is-h100-<tag>/leaf" ‖ 0x00 ‖ sub(0, 0) ‖ sub(0, 1)
          ‖ … ‖ sub(3, 3))[:32], so a 64 × 64 leaf holds 16 sub-digests and a proof of one subtile (#380's circuit
          tile) opens it with the other 15
tiles     64 × 64 (one m64n64k32 atom per slice: Round 7 measured 32 per word at N ≥ 64, and m64n8 at 22% of peak)
domain    64 | m, 64 | n, 32 | k, N ≤ m, k, n ≤ 2^16 (N = `mnk_min`: 8,192 by default, or any frontier size)
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
- **The D-3 family** (`D3Family(law, route, prices, sums)`), registered for every cell as
  `fp8-is-h100-<law>-v0[-y16][-f16]`. It is C-2's three NCP blocks formed on the unit's committed activation input,
  with one f16 noise field per side and per-entry amplitudes that cover one E4M3 cell of the entry and of its NCP
  partner.
  - **`law`:** **`d3s`**, the lead (Phase 17f), with amplitude max(|x|, |x_partner|, rms′)/8, δ = 1/8 and
    rms′ = max(rms, 2^−6) for all-zero rows. The alternative is `d3m`, with max(σ, cell(x), cell(x_partner)), δ = 1/16
    and σ ≥ 28/√k.
  - **`route`:** **`xq`** (the model's FP8 codes: the §0 input and Daniel's target) or `y16`, a comparison row
    (Y16 = f16(A/s) with the model's own row scale; k little-endian f16 words, finite and at most 448).
  - **Weights (Daniel, 20:31Z: no extra weight copies, FP16 or f32).** Each unit decodes the offline E4M3 codes and
    forms their statistic and amplitudes as the activation side does. So on `xq` the two sides' traces are equal.
    - Nothing per element is registered: an amplitude table is one f16 per weight element, the size of the withdrawn
      FP16 copy.
    - Keeping each column's rms′ between units (one word per column) would save 34.46 per weight element. It is a
      deployment requirement, so it is left to Daniel, not assumed.
  - **`sums`:** the format of the block values h + e_b.
    - `f32` (schedule M4): f32(h) + f32(e_b), rounded once, then `cvt.rn.satfinite.e4m3x2.f32`. It costs a widen of h
      shared by the three blocks, then a widen of each e_b and an FADD.
    - `f16` (schedule M2): HADD2, then `cvt.rn.satfinite.e4m3x2.f16x2`.
    - Either way a block value at or beyond ±464 saturates to ±448. The two differ by double rounding, so they are
      different checked functions with different tags.
  - **The streams.** E₁'s row i and F₁'s column j are SHAKE256("verity/pouw/fp8-is/<tag>/E1" or "/F1", salt, index,
    root_A, weight, i or j). They are per unit and per column (r = 1, X-D3-3).
    - Both read com(A_u), so no stream that forms the unit's operands exists before its activations are committed.
    - Without that, X-R9-2 applies: with F₁ known from the salt, X-D3-3 recurs at r = 1, ε ≈ 6.8%.
    - #218's lifecycle order is unchanged.
  - **The price** is the cell's per-element trace (`d3_trace`) from one `PRICES` row, which prices the forming layer and
    the core's FADD together (`get(name, prices=…)`). `FP8IS` rejects a layer priced from another row.
  - **The rows.**
    - **`round11`**, the default: `round10` with the centre at Round 11's 64.00 (§18.4). That is the split form as
      compiled, with c in a register, and it issues like the immediate form.
    - **The step rows** `round11-pipe` and `round11-dflow` price FADD at the §16.3 sweep's prediction for each honest
      step kernel's loop (`STEPS`):
      - `pipe` (n128 A = 2 U = 4, as built): a 567-instruction loop, FADD 32.0495, measured at 55.44 per add word;
      - `dflow` (n64 A = 5 U = 2, the short loop): 371 instructions and bank-clean, FADD 32.0118, measured at 52.67 per
        add word (§18.3, one run).
    - **`slowdown(forming, shape, step)`** is Track C's real-time model: T − 1 steps at the measured units per add word,
      plus one atom, plus the forming at class prices with no overlap, all over a plain FP8 GEMM's time.
    - **`round10`**: Round 10 (`gpu-constants.md` §16, ptxas 12.9), and the red team's reading of it.
      - FADD at its class price, 32.00, which holds for honest loops of at most about 390 instructions with
        bank-clean SASS.
      - `UNPACK_B` 64.17 per two codes, so the unpack (`.H0` and `.H1`) is 32.1 per code.
      - The PRMT-free four-code cast is 129.27 per register, 32.3 per code.
      - `VHMNMX` is 66.50 per two elements, HMNMX2 64.13 and the f32-pair F2FP 64.61.
      - The centre HFMA2 is 64.02 per two elements, the only form Round 10 timed (both constants immediate, unsplit).
      - HADD2 36, LOP3 64, HMUL2 a·K 32.17, HADD2.F32 64 and the f16 pack 64 carry over from Round 9.
    - **`round10-built`**: the same at FADD 32.05, today's 567-instruction honest loop, which is past the knee.
    - **`round9`** (Round 9 as measured, FADD 32.11) and **`round9-sass`** (ptxas 12.9's code at Round 9's classes,
      with the centre split at 36), kept for comparison.
    - Each statistic lane chains min(256, k/64) entries (`units_at(k)`), so rows below k = 16,384 pay more per element
      for the widen and the FADD.
  - **γ over the reporting frontier** (Daniel, 20:41Z), N³ with r = 1. The figures are computed from class prices;
    the honest D-3 kernel has not been run at any size:

    | Cell | Row (FADD) | 2,048³ | 4,096³ | 8,192³ | 16,384³ | 32,768³ |
    |---|---|---:|---:|---:|---:|---:|
    | **`d3s-v0-f16`** (X_q, f16 sums; the target) | `round11` (32.00) | 5.688% | 3.030% | 1.656% | **0.957%** | **0.605%** |
    | `d3s-v0-f16`, the short loop | `round11-dflow` (32.0118) | 5.705% | 3.048% | 1.674% | **0.975%** | **0.623%** |
    | `d3s-v0-f16`, as built | `round11-pipe` (32.0495) | 5.757% | 3.103% | 1.731% | 1.033% | **0.681%** |
    | `d3s-v0-f16` | `round10` (32.00) | 5.689% | 3.030% | 1.656% | **0.957%** | **0.605%** |
    | `d3s-v0-f16` | `round10-built` (32.05) | 5.758% | 3.104% | 1.732% | 1.034% | **0.682%** |
    | `d3s-v0` (X_q, f32 sums) | `round10` | 9.834% | 5.266% | 2.818% | 1.550% | **0.904%** |
    | `d3s-v0-y16-f16` (comparison) | `round10` | 5.688% | 3.030% | 1.656% | **0.957%** | **0.605%** |
    | `d3m-v0-f16` | `round10` | 6.152% | 3.276% | 1.782% | 1.021% | **0.637%** |
    | `d3s-v0-f16` | `round9` (32.11) | 7.658% | 4.160% | 2.322% | 1.380% | **0.903%** |
    | `d3s-v0-f16` | `round9-sass` (32.11) | 5.604% | 3.067% | 1.758% | 1.093% | **0.758%** |

    - **The target** clears at FADD 32.00 with threshold 32.028 (every FADD, core and forming, at the class price). As
      built, at 32.05, it is 1.034%, so it needs the honest loop under the knee.
    - **This reproduces the red team's Round 10 re-issue** in all 40 D-3 cells, to rounding.
    - **Nothing clears below 16,384³.**
    - **The centre is settled** (Round 11): with c in a register it issues at 64.00 per register, not the 36 of
      Round 8's two-source class.
    - **The slowdown over a plain FP8 GEMM** (`slowdown`, a prediction):

      | Step | 8,192³ | 16,384³ |
      |---|---:|---:|
      | as built (`pipe`) | 5.280× | 5.239× |
      | the short loop (`dflow`) | 5.021× | **4.979×** |

      At 16,384³ the short loop clears 5× and 1% together. 8,192³ needs 52.45 per add word for 5×.
    - **γ depends on m and n; k drops out** (under 0.004 points over the frontier's k).
    - `mnk_min` selects any frontier size as the domain's floor.
- **The two-level leaf's cost** (Round 11's SHAKE256, 2,290.6 units per byte from registers, per Keccak-f block of
  136 bytes). The combining hash is 5 Keccak-f per 64 × 64 leaf: 0.046× a plain GEMM at 8,192³ and 0.023× at 16,384³.
  Splitting each leaf into 16 digests adds 5 and 18 more Keccak-f per leaf (their domain strings and padding), 0.046×
  and 0.084×. Together that is 0.093× and 0.107×, which is +0.005–0.006% of hashing the words themselves (about 1,717×
  a plain GEMM, 0.75 bytes per MAC). The kernel, the checked words, W_ref and γ don't change. The step's slowdown
  excludes the commitment, as before.
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
- **Prices.** `PRICES` holds Round 7 as measured, Round 8 loop-free, Round 9 (`gpu-constants.md` §14), `round9-sass`,
  and Round 10 (§16: `round10`, the default, and `round10-built`). Rounds 7 and 8 carry the red team's assumptions for
  the X_q unpack and HADD2.F32.
- **Vectors.** `tests/vectors/fp8_is.json` (from `tests/fp8_is_vectors.py`):
  - every layer on a test instance;
  - `d3-v0` and `d2-draft` at k = 8,192;
  - the lead cell `d3s-v0-f16` at every frontier size, 2,048³ to 32,768³, one output cell each (bit-exact, CPU);
  - a full 64 × 64 tile of `d3s-v0-f16` (k = 128), with its 16 sub-digests and the leaf.

  Every case records its sub-digests beside the leaf.

  The core is checked against the total FP8 step and numpy's float32 running sum, and D-2's exact algebra (before any
  rounding) against A·Bᵀ.

## `ncp-v2`: NCP-INT as a circuit under sampled proofs (`verity_pouw/circuit`)

Sampled proofs is PoUW's verifier here: the call is a `verity.ir` Program, drawn units are proved, never recomputed, and
the audit's integrity profile carries the work bound (Project design note `docs/pouw/sampled-proofs-circuit.md`, §12; its
Lean statements are the store's `lean/submissions/pouw-accountable-compute`). This package has NCP-INT at small shapes
(16 | m, 16 | n, 64 | k); the certificate's domain is still `ncp-v1`'s.

**The units.** A weight's Y strips and c₀ read only B, the salt, the weight id and the column ids, so they are formed once
per run and weight: `NcpWeight{K,N,PERM}`'s root body is its `NcpFormY` units (`partition.weight_query`). One linear call
is `NcpLinear{M,K,N,PERM}`, which reads its weight's Y strips and c₀ as committed inputs. Each root body holds only
template instances, so each partition is `Q_template_instances` over them (`circuit.partition.query`); every value is
computed in exactly one unit. `NcpWindow{M,K,N,PERM,C}` is C calls on one weight in one root, the Y units once
(`partition.window_cut`): the composition passes the invariant too (`test_calls_on_one_weight_compose_without_recompute`).

~~~text
template            per                          reads                                         returns
NcpFormY{K,PERM}    16-column weight strip       16 weight rows, salt, weight id, column ids   the Y strip (s8), c₀
  (NcpWeight: once per run and weight)
NcpDigest{K}        16-row strip of A            its 16 A rows                                 D_s
NcpNode             node of the call's tree      its two children                              the parent (the root is D_A)
NcpKey              call                         D_A, salt, forward, position, weight id       K
NcpFormX{K,PERM}    16-row strip of A            16 A rows, K, row ids                         the X strip (X + β, u8)
NcpTile{K}          16 × 16 block                its X strip, its Y strip, c₀                  the credited words, and Z
NcpDequant          output entry                 Z, s_i, t_j, bias_j                           y = bf16(fp32(s_i)·fp32(t_j)·Z) + bias_j
~~~

**The formats.** Every hashed field is a 64-bit word: big-endian for SHA-512 (a digest word is its big-endian value) and
little-endian for SHAKE256. Tags are 8 ASCII bytes. `noise(b) = (b mod 64) − 31`.

~~~text
r_i = SHA-512("ncp2-row" ‖ row i's int7 bytes)            D_s = SHA-512("ncp2-stp" ‖ r_16s ‖ … ‖ r_16s+15)
D_A = RFC 6962 tree over D_1 … D_(m/16), node = SHA-512("ncp2-nod" ‖ left ‖ right); one leaf's root is the leaf
K   = SHAKE256(D_A ‖ "ncp2-key" ‖ salt ‖ forward ‖ position ‖ weight id)[:32],   position = layer << 32 | site
E₁ row i    = noise(SHAKE256(ι(i) ×5 ‖ "ncp2-E1\0" ‖ K)[:k])
F₁ column j = noise(SHAKE256(ι(j) ×5 ‖ "ncp2-F1\0" ‖ salt ‖ weight id)[:k]),        ι(i) = i << 32 | i
~~~

X, Y, c₀ and the checked values are `ncp-v1`'s (`schemes.ncp.assemble`, `checked_route_u`) on this noise: the tile returns
checkpoints 1 … 3k/16 − 1, step-major and row-major within a step, and Z, the last (A·Bᵀ exactly, mod 2^32). A and the
weight are committed a leaf per row, so a proved unit opens exactly the rows it reads.

**Why the formats look this way.** Each is forced by the partition, and each is checked by a test.
- **D_s is its own unit.** An IR node is one call, placed after its arguments, so one unit can't both write D_s and read
  D_A, which depends on it. The design note's X unit is two here: the digest unit and the forming unit. Both read the
  strip's rows, which are committed inputs, and they compute different functions.
- **K is per call, and each row's XOF begins with the row's own index in all five of Keccak's columns.** A row's XOF over
  (salt, forward, D_A, …, i) would compute round 1's values of the call-wide fields alone in every X unit: the same
  function of the same values in two units. The recompute check rejects that layout
  (`test_the_index_in_every_column_is_what_keeps_the_xof_recompute_free`). F₁ carries the column index the same way,
  and K's message begins with D_A's eight lanes, so that no two calls' key units compute a common value (X-SPC-71).
- **c₀ is the Y unit's.** Every tile of a column would otherwise compute −β·Σ Y[:, j] again.
- **The index binding is anchored inputs.** `Q_template_instances` admits no constant node at the root, so the forward
  index, the position, the weight id and the global row and column indices are Program inputs. Each has an anchor the
  verifier owns (`circuit.anchors`, the red team's X-SPC-72):

  ~~~text
  salt        the run's salt
  forward     the verifier's own enumeration of the window's forwards
  position    (layer, site) from the registered layout
  weight id   the registered id of the matrix whose rows B opens under the weights root
  row ids     the call's global row indices, distinct
  column ids  the weight's global column indices, shards included, distinct
  ~~~

  The verifier builds these inputs itself (`Traced.call_args`, `Traced.weight_args`) or refuses a prover's that differ
  (`Anchors.check_call`), and a window's `Ledger` refuses a repeated call index (forward, position), A11's
  `UniqueCallIndices`.
- **Every primitive returns at most 32 bits.** SHA-512 words are carried as (hi, lo) halves and Keccak lanes as
  (lo, hi).

**The verifier builds the Program** (X-SPC-73). `traced_as(scheme, m, k, n)` gives the call's and the weight's
Definitions from the scheme and the shape alone; the scheme pins P (`ncp-v2`: the word-granular P, as `ncp-v1`;
`ncp-v2-shift24`: the shift by 24). The verifier never takes a prover's descriptor: `Traced.check_descriptor` refuses one
whose digest isn't its own Program's. So the width rule is a property of that Definition, checked on it
(`partition.width_rule(shape, kind)` takes no Definition and no served count; a test's variant goes through
`width_rule_of`, which refuses an empty served set).

**The separator** (decision 4: option (a), Z stays in the tile). The protocol pins the separator port: the tile's Z, the
second field of `NcpTile`'s return (`partition.SEPARATOR_PORTS`). The registrant can't choose it.
- `partition.width_rule` removes the pinned ports' values, then width-checks every unit from which a served output (y)
  is still reachable, under `cut.WIDTH_RULE`. Only the dequantization units are, and each returns 16 bits. Every other
  unit is behind the separator.
- A tile whose word leaf reached y would itself be reachable, and would fail, along with every PoUW unit upstream of it
  (`test_a_word_leaf_wired_to_the_served_output_fails_the_width_rule`).
- `partition.separator` checks the topology. Every path from a PoUW unit (digest, node, key, form-X, tile; and form-Y in a
  window) to y passes through a pinned Z. What crosses is m·n values of 32 bits, 8,192 bits per tile, and no credited word is
  delivered.

**The draw** (`plan`). The strata are the verifier's own, as `verify` derives them (root's ruling on X-SPC-84): one per
template of the call's partition, in the order of its first unit (`Flock/Draw.lean` `strataOf`). The law draws each
stratum, and each drawn tile is proved with its closure: its X unit, strip digest, the node units on that strip's path and
the key, each once, and its Y unit in the weight's Program (`Plan.proved_y`), once per plan. Every other template is drawn
at its floor, so a wholly wrong one is caught directly (`audit_work_whole_stratum`). The law is a seam:
- `WorkLaw` is Verity root's work law (#362 with #374's floors, `Flock/Draw.lean`): k_s = min(n_s, max(f_s,
  ⌈K·w_s·n_s/W⌉)), with K = 27,713 at ε = 0.1% and δ = 2⁻⁴⁰ (`plan.record_k`; 27,712 misses). Only the tile template does
  PoUW work, so W is its stratum's, and k_tile = min(n_tile, max(f_tile, K)).
  - `plan.work_table` is the verifier's `--work-table`: every template of the call, `{"work": w, "floor": f}`, the tile's
    W_ref per tile and 0 for `digest`, `node`, `key`, `form_x` and `dequant`; `dequant`'s floor is y's (below), and the
    others' 1 unless POUS sets them. A call's work is in one stratum, the tile template's: `work_table`, `WorkLaw` and
    `window_laws` refuse any other (`plan.check_work_strata`), since the window pin's `covers_window` takes it as its
    hypothesis `hone` (with the cap K_c ≤ N_c, a second work stratum can be under-drawn).
  - `plan.law_object` is the law `verify` derives from it, each stratum `{template, units, work, floor, k}`.
  - K is a **window's** budget (X-SPC-83): a call draws its share K_c = ⌈K·W_c/W_window⌉, capped at its units
    (`plan.window_laws`). That is the law `verify` enforces when each call is its own statement with its own K_c
    (`--work K_c`); within one Program (`ncp2.window`) the calls share their templates' strata and one K.
  - **y's floors** (root's option B, which closes X-SPC-105). No tile's closure holds a dequantization unit, so y is
    drawn at its floor: each call's table gives `dequant` `{"work": 0, "floor": f_c}`, f_c = ⌈K_y·n_c/N⌉, n_c its y
    cells and N the window's, with K_y = K = 27,713 (`plan.window_laws`). So k_c = min(n_c, f_c) meets k_c·N ≥ K_y·n_c
    or proves the call's y, and a window draws at most K_y + C y cells. δ stays 2⁻⁴⁰: a tile-bad and a y-bad wrong set
    are one family of the one window audit, so its sampling term is the larger of (1 − ε)^K and (1 − ε)^K_y, not their
    sum. Per call the floor is pinned (`audit_work_floor`); `flock-verify draw --work` derives the same law from these
    tables.
  - **The window's bound is pending.** Both the per-call split of the tiles' K (X-SPC-83, 106) and B's window claim, an
    escape of at most (1 − ∑_c b_c/N)^K_y for b_c wrong y cells in each call (weighted AM–GM), wait on root's
    window-composition pin, `FlockSoundness/Audit/Window.lean`, which root's work-law lane drafts. It takes the per-call
    draws as independent, which the draw below gives.
- `CountLaw` is `main`'s stratified rule, k_s = max(1, round(K·n_s/N)).

**The draw is per call** (X-SPC-96). Each stratum's subset is drawn from the verifier's key under the call's index,
(forward, position), with the template: the strata are the call's own, one per template, and the calls of a window drawn
under one key draw independently, so K split across them keeps the window's (1 − ε)^K. The verifier draws through
`Traced.draw(key, law, anchors)`, which takes no layout: it draws over `Traced.layout()` (X-SPC-97), and keys each
stratum's subset with ("call", forward, position, "stratum", template), a context of the call's own, since a window's
`Ledger` refuses a repeated call index.

**The call's Y is linked to the weight Program** (X-SPC-95). The call reads its weight's Y strips and c₀ as committed
inputs, so at each drawn tile's column strip (the strips `Plan.proved_y` proves in the weight Program) the plan opens
the call's Y and c₀ there beside the weight Program's output there, a ("y", strip) leaf, and `Traced.check_link` refuses
any strip where they differ (`test_the_call_s_y_is_linked_to_the_weight_program`: another weight's Y, or one c₀ changed).
A wrong Y input on a strip no drawn tile reads spoils only undrawn tiles, which the law charges like any unsound tile.

**The layout is the verifier's.** The Layout theorems (`Layout.accountable_compute` and the rest) hold only for the
window's own layout, so the verifier derives each tile's own unit and its closure from the shape alone, through the
Program it builds (`Traced.layout`), and refuses any other (`Traced.check_layout`). `draw` refuses a layout whose tile is
not a tile unit or is not its closure's first unit: a tile mapped to its X strip is C1 again
(`test_a_prover_s_layout_mapping_a_tile_to_its_x_strip_is_refused`).

The profile is `Stratified` over the tile strata, and its `harm_bound`, at W_ref per tile, bounds the unsound work.

## NCP-FP8 (`fp8-is-h100-d3s-v0-f16`, `-d3s-v0`) as a circuit under sampled proofs (`verity_pouw/circuit/fp8.py`)

The same layout (a) as `ncp-v2`, over `fp8-is-*`'s D-3s forming on X_q (above): #295's scheme is the reference, and the
Program evaluates to it bit for bit (`circuit.fp8_reference`: root_A by `audit.commit_rows`, the forming, words and useful
output by `FP8IS(D3s(sums))`'s own `encode`, `checked` and `useful` on 16 × 16 tiles). `SUMS` is "f16" (the target cell) or
"f32". Its shapes: 16 | m, 16 | n, 32 | k with k admissible (π passes the d = 32 rule from k = 96).

**The units.** A weight's h and amplitudes read only its codes, so they are formed once per run and weight:
`NcpFp8Weight{K,N,SUMS}`'s root body is its `NcpFp8Amp` units (`partition.fp8_weight_query`). `NcpFp8Linear{M,K,N,SUMS,IL,OL,WL}`
is one linear call, which reads them as committed inputs; its root body holds only these templates' instances.
`NcpFp8Window{…,C}` is C calls on one weight in one root, the weight's units once, and it passes the invariant
(`test_calls_on_one_weight_compose_without_recompute`).

~~~text
template                     per                          reads                                        returns
NcpFp8Amp{K,SUMS}            16-column weight strip       its 16 weight rows (E4M3)                    h (f16; FP32 under f32 sums), the amplitudes
  (NcpFp8Weight: once per run and weight)
NcpFp8Domain{M,K,IL,OL}      call                         its prefix, index                            root_A's domain id, the tree's frontiers
NcpFp8Digest{K,RL,L0..L3}    16-row strip of A            its 16 rows (E4M3), row ids, frontiers       its 16 leaves' subtree root
NcpFp8Pad{RL,L0..L3}         16 pad leaves                pad ranks, frontiers                         their subtree root
NcpFp8Node{L}                tree node above the strips   its children, its (level, index)             the parent; the top is root_A
NcpFp8Key{K,SUMS,IL,WL,M,N}  call                         its prefix, index, root_A                    E₁'s and F₁'s frontiers
NcpFp8FormY{K,SUMS,IL,WL,LL} 16-column weight strip       its h and amplitudes, column ids, F₁'s       the Ỹ strip (16 × 3k codes)
                                                          frontier
NcpFp8FormX{K,SUMS,IL,WL,LL} 16-row strip of A            16 rows, row ids, E₁'s frontier              the X̃ strip (16 × 3k codes)
NcpFp8Tile{K}                16 × 16 block                its X̃ strip, its Ỹ strip                     the words, and Z
NcpFp8Dequant{BIAS}          output entry                 Z, s_i, t_j (and bias_j)                     y, in vLLM's epilogue order
~~~

- **root_A** is the protocol's commitment to A's rows: `verity.commitments.merkle`'s SHA-256 frame-v3 tree, one leaf per
  row (the row's k codes), its domain bound to `activation_binding(salt, index)` with owner = index, padded to a power of
  two. Strips are its aligned 16-leaf subtrees.
- **E₁ and F₁** are `D3Family.stream`'s SHAKE256 streams over (salt, index, root_A, weight id, line), 2k bytes each.
- **The served output** is vLLM's `cutlass_scaled_mm` epilogue, bit for bit, as `H-1T`'s is (`NcpFp8Dequant{BIAS}`, as
  the vLLM integration's exact-tested `ScaledMmFp8Coordinate` pins it): m = RN(s_i·Z), the row scale first under the
  batch-invariant dispatch's swapped scales, then y = cvt.rn.bf16(RN(t_j·m + f32(bias_j))) in one FFMA, or
  cvt.rn.bf16(RN(t_j·m)) for a linear without a bias (its own Definition, `NcpFp8Linear{…,BIAS=False}`: the two differ
  on a −0). `traced_as(…, bias=…)` picks it.
- **The tile** runs `cell_words`: T = 3k/32 fresh E4M3 k32 atoms from +0 (`HopperE4m3QgmmaDot32`) and the running FP32
  sums (`F32Add_v1`). Per cell its words are I_1 … I_T and R_2 … R_(T−1), and Z = R_T, the useful output, is field 1.
- **The arithmetic** is #295's `verity.ml.tc.fp16` and `fp32`, one primitive per operation (`circuit.fp8_words`), and
  core's FP32 add, mul and div, which equal #295's on every non-NaN word.

**Hoisting** (`circuit.hoist`). Every stream's message and every leaf's and node's hash share call-wide prefixes (the salt,
the index, root_A and the weight id; the commitment domain). Computed in each row's unit, those gates would be the same
functions of the same values in many units, which `validate_unit_cut` rejects. So each per-row function is built once as a
DAG over inputs marked call-wide or the row's own, and split:
- its call-wide gates run once, in `NcpFp8Domain` (the tree's) or `NcpFp8Key` (the streams'), which return the frontier,
  the call-wide values a row's gate reads;
- its row gates are a Definition of their own (`NcpFp8Leaf{K,RL}`, `NcpFp8StripNode{DEPTH,L}`, `NcpFp8UpperNode{L}`,
  `NcpFp8PadLeaf{RL}`, `NcpFp8XRow` and `NcpFp8YRow{K,SUMS,IL,WL,LL}`) that reads the frontier as committed values.

A hash block whose inputs are all call-wide or all the row's own is one call of its Definition (`PouwKeccakF1600`,
`PouwSha256Compress`); a mixed block is expanded into primitives. Without the split, two rows recompute the streams'
prefix (`test_without_hoisting_two_rows_recompute_the_streams_prefix`).

**The prefix** (X-SPC-90). The domain's and the key's gates over the salt and the weight id alone would compute the same
values in every call of a run, or of a weight: `activation_binding`'s first block and rounds (901 gates) and the streams'
first-block lanes that hold only the tag, the salt and the weight id (69 per stream). They are functions of the verifier's
anchors, so the verifier evaluates them natively (`fp8.prefix`, `hoist.public_part`) and the call reads the values its
remaining gates need as one anchored input, `prefix`; the call has no salt or weight id input of its own. A Ỹ column's part
over its codes alone (the decode, the rms, the floor and the amplitudes: 35–39% of its forming's ANDs, by the red team's count) is its weight's `NcpFp8Amp`;
F₁ is keyed per call on root_A, so the rest of the column stays in the call.

**Index binding.** The prefix, the index, the row ranks, the rows' stream lines, the column lines, the pad ranks and the
upper nodes' (level, index) are Program inputs. The verifier builds every one (`anchors.Fp8Anchors`, X-SPC-89): the prefix
is its own evaluation from the salt and the weight id, the index is the call's declaration, and the rest are functions of
the shape (ranks and lines 0 … m−1,
column lines 0 … n−1, pad ranks m … width−1, `fp8.upper_nodes`), so y is fixed by the committed A, B, the salt and the
declaration alone. `Traced.call_args` builds them, `Fp8Anchors.check_call` refuses a prover's that differ, and a window's
`Ledger` refuses a repeated index. A row's rank (the tree's) and its line (E₁'s) are equal but separate inputs, equal by
construction: read as one, the digest and the X unit place its bytes by the same gate once they take two bytes, and that
value would be computed in two units. The encodings are minimal big-endian (`frame`, `merkle._uint`), so byte lengths are
template parameters, and the anchors fix the call's:
- per call, the index's (IL), index + 2's (OL, the domain's owner) and the weight id's (WL), `fp8.Lengths.of(index, weight)`;
- per binding, a strip's ranks and its nodes' indices (`fp8.strip_lengths`), an upper node's index, and a row's or
  column's line. Past 255 they take two bytes, so strips are batched in runs of one byte-length signature, and the domain
  and key units return one frontier per variant (`fp8.domain_layout`, `fp8.key_layout`).

**The separator and the width rule** are `ncp-v2`'s, read through `partition.Circuit`: the pinned port is the tile's Z
(field 1), every path from a PoUW unit (the domain, digests, pads, nodes, the key, the forming, the tiles) to y crosses a
Z, 8,192 bits per tile, and only the dequantization units (16 bits each) are width-checked.

**The leaf** (X-SPC-92). #295's transcript leaf is two-level, its production default: a 64 × 64 tile's leaf is
SHAKE256 over its 16 subtiles' digests, and a subtile is the circuit's 16 × 16 tile. From a tile's credited words (its
first field: cell-major, per cell I_1 … I_T and R_2 … R_(T−1)) and each cell's R_T = RN_f32(R_(T−1) + I_T), the tile's own
`F32Add_v1`, `fp8.subtile_digest` and `fp8.leaf_digest` give #295's bytes:

~~~text
sub(a, b) = SHAKE256("verity/pouw/fp8-is-h100-<tag>/subtile" ‖ 0x00 ‖ the layer's words ‖ the core words)[:32]
            the layer's words: its rows' then its columns' (none for D-3s); the core words: little-endian, step-major,
            row-major over the subtile's cells within a step, I_τ before R_τ, R_1 = I_1 unchecked, R_T = Z included
leaf      = SHAKE256("verity/pouw/fp8-is-h100-<tag>/leaf" ‖ 0x00 ‖ sub(0, 0) ‖ sub(0, 1) ‖ … ‖ sub(3, 3))[:32]
            subtile (a, b) is rows 16a … 16a + 15 and columns 16b … 16b + 15 of the 64 × 64 tile, row-major
~~~

`test_the_subtile_digests_are_295s_leaf_vector` holds this to #295's pinned `leaf-64-d3s-f16` (all 16 sub-digests and the
leaf), and `test_the_circuit_s_tiles_give_295s_production_leaf` holds a 64 × 128 × 64 call's circuit tiles to #295's
production leaf on the same call.

**The draw** (`plan.fp8_layout`). A tile's closure is `ncp-v2`'s (its X and Y strips, its strip's digest, the nodes on
the strip's path, the key) plus the domain unit and the **pad cone**, every pad strip and every node with no row beneath
it: a pad has no tile beneath it that a wrong one could be charged to. Its Y strip's `NcpFp8Amp`, in the weight's
Program, is proved once per plan (`Plan.proved_y`), and opens the strip's weight rows. A tile's W_ref is its cells' share of
`FP8IS.wref`. The strata are one per template, as for `ncp-v2` (every byte-length variant its own), the work table
gives the tile template its W_ref and every other template 0, floors 1 but y's: in a window each call's
`NcpFp8Dequant` has option B's floor ⌈K_y·n_c/N⌉ (`window_laws`, as for `ncp-v2`). `WorkLaw()` draws every layout: K is a
window's, and a call alone is capped at its units (X-SPC-83). Each call's subsets are keyed by its matmul index with the template, as
`ncp-v2`'s (X-SPC-96), and the verifier draws through `Traced.draw`. The call's h and amplitudes are linked to the weight
Program as `ncp-v2`'s Y is (X-SPC-101): the plan opens them at each drawn tile's column strip beside the weight Program's
outputs there, and `Traced.check_link` refuses a difference. A window (`NcpFp8Window`) serves every call's y, C·m·n
outputs (`partition.Circuit.calls`), and `anchors.fp8_window_inputs` builds its per-call anchored inputs from each call's
`Fp8Anchors`, refusing a repeated index or calls of another salt or weight id (X-SPC-102, 108); its separator reads
every call's y too (X-SPC-109). A window form of `traced_as`, with the window's layout and draw, is not here yet. The
layout is the verifier's, as `ncp-v2`'s: derived from the shape and the declaration's lengths, and any other refused
(`Traced.check_layout`).

## Not here yet

- The fast implementations: the RTX 4090 kernels and the vLLM adapter (`integrations/vllm/verity_vllm/protocol_options/pouw.py`)
  are gated by these vectors when they land.
- The per-weight share of the honest reference's work in `wref`, and a `TargetProfile` Definition of the NCP GEMM.
- The honest D-3 kernel on an H100 (its forming loop and the checked step together): not run at any frontier size.
  Round 10 timed the classes and the checked step's harness shapes, not this kernel. The γ figures come from class
  prices.
- For `ncp-v2`:
  - the lead domain's shapes (k ≥ 1,088) as a recorded run rather than a test;
  - C-Flock pieces for the PoUW primitives;
  - the floors POUS sets for the zero-work templates other than `dequant` (X-SPC-55; 1 until then);
  - root's window-composition pin (`FlockSoundness/Audit/Window.lean`), for the per-call split's window-level bound
    (X-SPC-83, 106) and B's window claim for y (X-SPC-105, closed by option B per call);
  - the served-output designation and the pinned ports in `Q_nested_instances` v0's spec, evaluator and Lean port;
  - the vLLM adapter's `traced_as`;
  - the reuse of a weight's Y proofs across a run's windows (X-SPC-51), which needs its commitments linked across windows;
  - H-1T.
- For NCP-FP8's circuit:
  - production shapes (k in the thousands, m and n up to the model's) as a recorded run;
  - C-Flock pieces for its primitives;
  - the leaf's link in the Program: a unit that hashes a drawn tile's words to its sub-digest and combines it with the
    opening's 15 siblings (committed inputs) into the transcript's leaf, and its place under the separator and the
    width rule. Today the tile returns its words unhashed, and `fp8.subtile_digest` computes the sub-digest from them.
  - a window form of `traced_as`: the verifier's `NcpFp8Window` Program, layout and draw over its C calls. Its width
    rule, separator and anchored inputs (`fp8_window_inputs`) already cover every call.
