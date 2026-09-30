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

## `pearl-c-h100-v0` and `pearl-c-h100-v1`: Pearl-C, Pearl's product with its noisy clean-up checked (`schemes/pearl_c.py`)

The cheap-binding lane's H100 candidate (research store `docs/pouw/cheap-binding.md` §1). It takes Pearl's H100 product
and output tickets, and adds #311's noisy clean-up pinned bit for bit, so that the useful output U is a checked value.
γ rests on the named conjecture TT_OUT(1/400), not on counting, so the certificate names no γ_0. The module's docstring
states every step.

~~~text
seeds     seed_B = H_"pearl-c/v0/seed-B"(salt ‖ root_B);  seed_A = H_"pearl-c/v0/seed-A"(salt ‖ root_A ‖ root_B ‖ index u64)
forming   v0, #311's as ported (`pearl_kw`): FP10 rows, exact norms, α, β, N = E·Fᵀ, X̃ = fp8(clamp(fma(α, X, β·N)))
          v1, §5's: FP32 rows x; s = max|x|, ρ = RMS over every 8th position, α = 448/max(s + 4ρ, 2^−32), σ = ρ,
          β_row = (σ·α)·√r/(256·16), E′ = e4m3(β_row·E), X̃ = e4m3(HOPPER_E4M3_K32.step(C = α·x, E′_row, F line)),
          with F_A and F_B's lines at L2 norm 16 (E's at Pearl's 256)
chain     C̃ = the H100 replay of A′·B̃ᵀ: a +0 accumulator per 128 products chains four HOPPER_E4M3_K32 atoms,
          promoted by one FP32 nearest-even add (−0 → +0)
peel      P_A = [bf16(β_A·E_A) ‖ bf16(A′·F_B)],  P_B = [bf16((β_B·E_B)·G − B̃·F_A) ‖ −bf16(β_B·E_B)],  G = F_Bᵀ·F_A
          (v1: bf16(E′) in place of bf16(β·E));
          U = C̃ + P_A·P_Bᵀ as four HOPPER_BF16_WGMMA_K16 steps into C̃'s word
leaf      TurboSHAKE128 (D = 0x21) per tile row and 32-column chunk over its C̃ then U words, then TurboSHAKE128
          (D = 0x22) over those digests
tickets   Pearl's lottery over C̃ per 16 × 16 subtile
useful    U/(α_A·α_B), glue
~~~

- **Pinned where #311 isn't.** #311 runs G and the B-side factor on a tolerance path.
  - Here G, A′·F_B and B̃·F_A are the same FP8 chain as C̃ (A′·F_B is the product's 32 fused columns).
  - (β_B·E_B)·G is a sequential FP32 FMA chain from +0.
  - The peel is the BF16 wgmma accumulating into C̃'s register.
- **Conformance.**
  - The chain equals Pearl's own Rust H100 replay at `284b147b`: the pinned matmuls, and both 16 × 16 H100 tiles
    after `pearl_kw`'s forming (`tests/vectors/pearl_kw.json`).
  - It equals `pearl_kw.replay` on random codes, partial windows included.
  - The scheme's own vectors are `tests/vectors/pearl_c.json`: for each forming, a 16 × 16 × 128 test instance and one
    64 × 1,024 × 64 tile.
- **Two formings.**
  - v0 is the port, not the design: it keeps #311's FP10 decode and exact norms, whose A-only work the red team counts
    as skippable (X-PC-2).
  - v1 is the rework (`cheap-binding.md` §5), with its roundings pinned in the module docstring: the order of ρ's
    partial sums, α and β_row's FP32 steps, and the 2^−32 floor that keeps an all-zero row finite.
    - Only ρ depends on A alone. The model adopts s, ρ and α as its own FP8 quantizer, so W_ref counts 68 units per
      element of forming, 64 of them credited (the lane's `Costs.adopted`).
    - `wref` is Lean's `wrefOf` and `credit_of` its `creditOf`, pinned on a grid of shapes. The weight side (B's forming,
      B̃·F_A, G, P_B) is `job_cost`, once per weight and epoch, in neither.
    - Its domain is TT_OUT's: 128 ∣ k (whole promotion groups), and on every row of both operands the noise floor ρ·α ≥ 1
      (exact product) and liveness on the exact statistic, at most k/64 entries with x² ≥ 64ρ², where
      ρ² = Σ_{p ≡ 0 (8)} x_p² / ⌈k/8⌉ over the words' values (`live_row`).
    - It shares v0's seed labels (`pearl-c/v0/seed-*`).
- **The debit** (`schemes/pearl_c_debit.py`) is `Skip.lean`'s, replayed by the verifier on a drawn tile.
  - `word_flags` flags an atom when some set of its promotion group's atoms containing it, skipped jointly, leaves the
    word's output unchanged. The later promotions are tested by intervals found backward by bisection.
  - It equals P1's 552 Lean-checked word vectors (`tests/vectors/pearl_c_skip_vectors*.txt.gz`).
  - A tile's chain debit is 40 units per flagged atom-word, summed over its words. The forming debit's per-tile rule
    is pending from P3's owner, so `forming_elements` is a stub that returns 0.
  - The census's aligned-spike tile at R = 300 passes the domain with 74.6% of its atom-words flagged, about 15 times the
    cap.
- **The work law** (`schemes/pearl_c_work.py`) is what the verifier credits, draws and checks.
  - A weight's `Split` holds at most ⌊k/100⌋ side channels. The unit reads the kept channels zero-padded to `k_unit`,
    and credit is on the kept width, so padding earns nothing. The audit recomputes S by the pinned rule through
    `split_check`, which the model integration supplies.
  - Before the draws the prover declares a record per matmul: its real rows (the rest are zero filler), its kept width,
    and its excluded rows. `records_root` binds the records and the splits.
  - One exclusion rule, per row: a row that fails the per-row rules stays in the unit, bound as any other, and its cells
    earn no credit. Tiles with credited cells stay drawable, and the audit also opens one tile per excluded row.
  - A cell's draw weight is its row's share of `wrefOf` at the kept width. An accepted unit is credited
    (1 − ρ)·`creditOf` over its credited cells.
  - On an opened tile the verifier checks that filler rows are zero, that every row is zero past the kept width, that
    credited rows pass the per-row rules and excluded rows fail them, and that the replayed debit over the credited words
    is at most ρ·`creditOf` of the tile, with ρ = 1/400.
  - The draws are keyed by a beacon round published after the verifier recorded the commitment (the transcript root and
    the records root), through `derive`.
- **The TurboSHAKE128 reference** is `verity.commitments.turboshake`, checked against RFC 9861 and `hashlib`'s
  Keccak-f[1600].
- **U's exact order, CPU-pinned** (`schemes/pearl_c_u.py`), for the Lean model of the clean-up.
  - It states the chain with its promotion, the fused columns, the factors' single BF16 roundings, P_B's FMA chain and the
    four K16 steps, and both tensor-core steps as integer grouped sums.
  - It equals `pearl_c` bit for bit.
  - Its vectors (`tests/vectors/pearl_c_u.json`) are cells of the H100 kernel's gate problem, and they agree with the
    gate's expected buffers, whose hashes they record.
  - The H100 capture confirms it on silicon.

## `pearl-c-sm120-v1`: Pearl-C v1 on the RTX PRO 6000 Blackwell (`schemes/pearl_c_device.py`)

The same scheme as `pearl-c-h100-v1` on another device record. The record (`Device`) is a scheme parameter: the FP8 atom
(E4M3, k = 32), the BF16 peel step (k = 16), G (atoms per promotion), the W1 prices (`Prices`) and the domain's k bound.
`pearl_c`, `pearl_c_debit`, `pearl_c_u` and `pearl_c_work` read them from it, and the H100's record (`H100`) is every
function's default, so every `pearl-c-h100-*` output and vector above is unchanged.

- **The atom is the card's captured step**, `BLACKWELL_SM120_E4M3_M16N8K32`: `mma.sync` `kind::f8f6f4` m16n8k32 E4M3 as one
  group of all 32 products and the accumulator in a 26-bit adder (Lean's `Pipeline ⟨[32], 26, −133⟩`), where the H100's
  and Ada's FP8 adders are 14 bits wide.
- **Every other field of `SM120` is a stand-in**, and `standins` names each one; the certificate lists them as hypotheses.
  - The peel is `HOPPER_BF16_M16N8K16`, which matched the RTX 5090's `mma.sync` m16n8k16 BF16.
  - G = 4, the H100's W1 prices and k ≤ 2^16 are the H100's. They are statement parameters, changed through the
    coordinator and the cheap-binding and Lean lanes; P1 and the debit are not yet re-derived for a 26-bit adder.
- **`SM120_UNPROMOTED`** is the same record with G = None: one accumulator from +0 over the whole of k, whose word is the
  last atom's output, with no promotion adds in `wref` and 32 ∣ k. No scheme is registered on it, and the debit
  raises on it, until Pearl-C is restated without promotion.
- `pearl_c_u` restates a step of several groups as one grouped sum per group, the accumulator joining the first and each
  group's word the next.
- Only forming v1 runs on a non-H100 record (v0 is #311's H100 port).
- The work law aligns `k_unit` to 128, so it requires 32·G ∣ 128.

## Not here yet

- The fast implementations: the RTX 4090 kernels and the vLLM adapter (`integrations/vllm/verity_vllm/protocol_options/pouw.py`)
  are gated by these vectors when they land. Pearl-C's H100 kernel is `benchmarks/pouw/pearl_c/`, its sm_120 kernel
  `benchmarks/pouw/pearl_c_sm120/`.
- The per-weight share of the honest reference's work in `wref`, and a `TargetProfile` Definition of the NCP GEMM.
