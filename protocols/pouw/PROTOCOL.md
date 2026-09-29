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
    W_ref per tile and 0 for `digest`, `node`, `key`, `form_x` and `dequant`, floors 1 unless POUS sets them.
  - `plan.law_object` is the law `verify` derives from it, each stratum `{template, units, work, floor, k}`.
  - K is a **window's** budget (X-SPC-83): a call draws its share K_c = ⌈K·W_c/W_window⌉, capped at its units
    (`plan.window_laws`). That is the law `verify` enforces when each call is its own statement with its own K_c
    (`--work K_c`); within one Program (`ncp2.window`) the calls share their templates' strata and one K. The per-call
    split's window-level bound is X-SPC-83's question, and no pin states it.
- `CountLaw` is `main`'s stratified rule, k_s = max(1, round(K·n_s/N)).

**The draw is per call** (X-SPC-96). Each stratum's subset is drawn from the verifier's key under the call's index,
(forward, position), with the template: the strata are the call's own, one per template, and the calls of a window drawn
under one key draw independently, so K split across them keeps the window's (1 − ε)^K. The verifier draws through
`Traced.draw(key, law, anchors)`, which takes no layout: it draws over `Traced.layout()` (X-SPC-97).

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

## H-1T as a circuit under sampled proofs, layout A (`verity_pouw/circuit/h1t.py`)

Track H's tagged one-sided FP8 construction (docs `pouw/hardness-shaped-matrices.md` §3.1) as a Program under layout A,
the scope of the restated `TTH1T`'s grant (a named assumption for layout A at 16 × 16 tiles, k <= 32,768, the units,
tiling and Program fixed before the salt). There is no H-1T scheme in `schemes` yet: the reference is Track H's pinned map,
`h1_tagged.h1t_ref`, whose forming and chain `tests/vectors/h1t_ref.json` pins bit for bit (Track H's own code, run on raw
salt words in the honest kernel's bit layout). Shapes: 16 | m, 16 | n, 8 | k, 29 < k <= 32,768, n <= 88,412.

~~~text
template        per                   reads                                                     returns
H1tKey{K}       16-row strip of A     its rows, its strip id, salt, forward, position, weight id  K_s
H1tFormX{K}     16-row strip of A     its rows, their row ids, K_s                                [x1 | x2] per row (2 × 32T codes)
H1tTile{K}      16 × 16 block         its x strip, 16 weight rows (B′'s real codes), their tags   the credited words, and Z
H1tDequant{BIAS} output entry         Z, s_i, t_j (and bias_j)                                    y, in vLLM's epilogue order
~~~

- **The map** (`h1t_words`, one primitive per lane): T = ceil(k/29) slices of 29 real lanes (0 past k) and 3 tag lanes.
  Per slice, F = max(2^(e(M) − 6), 2^−2); on a real lane mu = max(p/8, F), x1 = enc(x + s·mu), x2 = enc(−s·mu); on tag
  lane i, v_i = s_i·32F·(1 + m_i/8), x1 = v_i, x2 = −v_i. The salt bits are the kernel's (`h1_kernel.h1t_form_xq`):
  per slice raw words r0 … r17, a real lane's bit 15 or 31 of r_(l div 2) set when x2 < 0; tag lane 29 from r17's bits
  31 and 25 … 23, lanes 30 and 31 from r16's 15 and 9 … 7, and 31 and 25 … 23.
- **B′** is the weight's registered E4M3 rows, laid out per slice with the column's fixed tag tuple (A0[j mod 23],
  A1[(j div 23) mod 62], A1[(j div 1426) mod 62]), an anchored input. There is no weight-side forming.
- **The served output** is vLLM's `cutlass_scaled_mm` epilogue, bit for bit (`H1tDequant{BIAS}`, as the vLLM
  integration's exact-tested `ScaledMmFp8Coordinate` pins it). The batch-invariant dispatch swaps the scales, so the row
  scale goes first: m = RN(s_i·Z). With a bias, y = cvt.rn.bf16(RN(t_j·m + f32(bias_j))), one FFMA
  (`ScaledEpilogueBias`); without one, y = cvt.rn.bf16(RN(t_j·m)) (`ScaledEpilogue`). The two differ on a −0, so a linear
  without a bias has its own Definition (`H1tLinear{…,BIAS=False}`), and the verifier picks it (`traced_h1t(…, bias)`).
- **The chain** per cell: I_1 … I_T over x1, then I_(T+1) … I_2T over x2, each a fresh E4M3 k32 atom from +0
  (`HopperE4m3QgmmaDot32`); R_1 = I_1 and R_τ = RN_f32(R_(τ−1) + I_τ). The credited words are I_1 … I_2T and
  R_2 … R_(2T−2), 4T − 3 per cell; Z = R_2T is the useful output and the pinned separator port, as in `ncp-v2`.
- **The salt stream is the circuit's**, tile-local as the FP8 red team's X-SPW-5 has it (D_s in root_A's place, and the
  global row index, X-SPW-2), in `ncp-v2`'s formats: r_i = SHA-512("h1t-row\0" ‖ row i), D_s = SHA-512("h1t-stp\0" ‖
  r_16s … r_16s+15), K_s = SHAKE256(ι(strip) ×5 ‖ "H1T-KEY\0" ‖ salt ‖ forward ‖ position ‖ weight id ‖ D_s)[:32], row
  i's words = SHAKE256(ι(row) ×5 ‖ "h1t-slt\0" ‖ K_s)[:72T]. Track H's reference fixes the map, the words' bits and the
  chain, not this derivation. The strip id and the row ids are separate anchored inputs.
- **The anchors are the verifier's** (`anchors.H1tAnchors`): `ncp-v2`'s fields (the salt, the forward index, the
  position, the weight id, the call's global rows and the weight's global columns), with each strip's id its rows' // 16
  (rows come in aligned strips) and each column's tag tuple `h1t.tag_tuple(j)`. `Traced.call_args` builds them and
  `H1tAnchors.check_call` refuses a prover's that differ.
- **The verifier builds the Program and the layout** (`circuit.traced_h1t(m, k, n)`, since H-1T has no registered scheme
  for `traced_as` yet), from the shape alone and only within the granted scope: layout A at 16 × 16 tiles, 16 | m,
  16 | n, 8 | k, 29 < k <= 32,768.
- **The partition** (`partition.h1t_circuit`): the whole-call `validate_unit_cut` passes, every path from a key, forming
  or tile unit to y crosses a pinned Z (8,192 bits per tile), and only the dequantization units are width-checked.
  `H1tWindow{M,K,N,C}`, C calls on one weight in one root, each with its own anchored indices, passes the invariant too.
- **The draw** (`plan.h1t_layout`): the strata are one per template, as `verify` derives them (root's ruling on
  X-SPC-84), the tile's with its W_ref and the others at their floors. A tile's closure is the tile, its X strip's
  forming unit and that strip's key. A
  tile's W_ref is its cells' 2T k32 atoms (`h1t.tile_work`, 64·T MACs a cell; the forming's share isn't counted). A
  proved tile opens its X strip's rows and its 16 weight rows. The layout is the verifier's: `Traced.check_layout`
  refuses any other, and `draw` refuses a tile mapped to its X strip. Each call's subsets are keyed by its (forward,
  position) with the template (X-SPC-96), and the verifier draws through `Traced.draw`. H-1T has no weight Program: its
  tiles read B′'s rows directly, so there is no Y link to open.

## Not here yet

- The fast implementations: the RTX 4090 kernels and the vLLM adapter (`integrations/vllm/verity_vllm/protocol_options/pouw.py`)
  are gated by these vectors when they land.
- The per-weight share of the honest reference's work in `wref`, and a `TargetProfile` Definition of the NCP GEMM.
- For `ncp-v2`:
  - the lead domain's shapes (k ≥ 1,088) as a recorded run rather than a test;
  - C-Flock pieces for the PoUW primitives;
  - the floors POUS sets for the zero-work templates (X-SPC-55; 1 until then), and the per-call split's window-level
    bound (X-SPC-83);
  - the served-output designation and the pinned ports in `Q_nested_instances` v0's spec, evaluator and Lean port;
  - the vLLM adapter's `traced_as`;
  - the reuse of a weight's Y proofs across a run's windows (X-SPC-51), which needs its commitments linked across windows;
  - NCP-FP8.
- For H-1T's circuit: an H-1T scheme in `schemes` (its stream derivation pinned), with `traced_as` routing to
  `traced_h1t`; the legal-input check (|x| <= 416) on committed rows; and W_ref with the forming's share.
