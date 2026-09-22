# Candidate A on the GPU: design

Track A's prover (GKR + LogUp + Ligero, `../PROTOCOL.md`) in torch (+Triton) over BabyBear^6, for the full
K=1536 VU (96 chained units + epilogue, PROTOCOL §12) at batch 4096 on one H100.  The CPU Rust crate
(`../src/`) is the reference for the *protocol*; this port keeps its message order, layer order, LogUp tree,
Ligero parameters and Fiat-Shamir structure, and changes the field.

## 1. Field, circuit, embedding

* Base field `p = 2^31 - 2^27 + 1` (BabyBear), challenges in `F_{p^6} = F_p[X]/(X^6 - 31)` (the same `W = 31` as
  `../tensor/reference.py`).  Tables are u32 BabyBear (int32 tensors, int64 compute); an extension element is six
  u32 in planar order `c0..c5` (a trailing dimension of 6).
* The circuit is the WP2 checker **under the red-team's BabyBear embedding** (`babybear_rules.md`): it is
  `verity_numerical.redteam.babybear.check_step_bb` run on the symbolic `CircuitCtx` of `gkr_export`
  (`bb_export.py`).  FP32 words are two public 16-bit limbs (`x.c_lo/x.c_hi`, `y_lo/y_hi`, epilogue `y32_lo/y32_hi`,
  rule LIMB16); wide variable bounds are the direct checks against the tables' bound columns (LT_DIRECT,
  QUOT_TIGHT, NORM_MBND, NORM_VBND).  Unit circuit: 452 columns, 774 wires (s_w = 10), depth 4, 326 assertions,
  609 lookups per unit (587 R16 + 2 POW + 16 ALIGN + 2 LEAD + 2 NORM); epilogue: 8 columns, 10 wires, 4 R16.
  Nothing of the relation is dropped: every predicate of the embedded checker is a gate, an assertion or a lookup.
* **Flagged deviation from `gkr_export.circuit.CircuitCtx`** (present in the Goldilocks export too): its
  sub-16-bit range check is one lookup `R16 ∋ y · 2^(16-w)`, which over a field accepts `y = r · 2^-(16-w) mod p` for
  every table entry `r` (the multiplication is a bijection of the field; `y` is not bounded).  `BBCircuitCtx`
  realises `rng(y, 0, 2^w)`, `w < 16`, as the two lookups `R16 ∋ y` and `R16 ∋ y + 2^16 - 2^w`
  (the first makes `y` a canonical integer `< 2^16`, so the sum does not wrap and the second bounds it).  This is
  a soundness fix, not a protocol change; it adds R16 queries (587 vs the Goldilocks export's count).  The CPU lane
  should apply the same fix.
* The chain (§12) links the *two limbs*: per VU `96·2` incoming/outgoing limb identities, 2 for the epilogue
  input, 1 for the public output word: `(96+1)·2 + 1 = 195` linear constraints per VU with geometric coefficients
  `ρ·geo^k` in the CPU crate's order (limb-major within a step).

## 2. Data layout on the device

* Segment 0 (units): `cols` `(B·96, 452)` int64 canonical, VU-major (`row v·96 + i`); segment 1 (epilogue): `(B, 8)`.
  Loaded from `units.bin` / `epilogue.bin` (u32 LE, `<QQ` header) written by `bb_export`.
* Wires: `(units, 774)` int32, evaluated once per segment by the gate list (each product wire = two linear forms
  of earlier wires, `circuit.eval_wires`).  The layer-`k` view zeroes wires of depth `> k` (`masked_wires`).
* Layers (`circuit.layers`, the Rust `layers()`): layer 0 = the 326 assertions over `V^(4)` (s_out = 9), then
  `depth4 .. depth1` each computing every wire of depth `<= k` from the previous view (s_out = s_in = 10), the last
  reading the columns (s_in = 9).  Each layer's wiring is two **dense** matrices `A, B` `(g_n, 2^s_in)` on the device
  (challenge-independent, built once, ~1024 x 1024 int64): `Ã_g(c) = Σ_l A[g,l] V_in(l,c)` is one `mm_mod` of the
  input rows with `A^T`.
* The batch dimension is the copy index `c` of the data-parallel GKR (PROTOCOL §12): `b = log2(B·96) = 19` at
  B = 4096 (2^19 copies, the 131072 pad rows are zero).

## 3. Round schedule (identical to `../src/main.rs`)

1. **Commit** `X = [unit columns row-major, epilogue columns, table multiplicities]` with Ligero (challenge-free).
2. Transcript preamble; per table `(z, β)`; per segment the output point `z0` (s_out challenges) and copy point `γ`
   (`b` challenges).
3. **LogUp**, one fractional-sum GKR per table (POW, ALIGN, LEAD, NORM, R16 — the R16 tree has `n = 28` at B = 4096:
   `4096·96·587 + 4096·4 + 65536 = 2.31e8` leaves): leaves `(1, z - Σ β^k v_k)` per query, `(-m_j, z - Σ β^k t_j)` per
   table row, `(0, 1)` padding; `p` leaves are base-field (stored as one int32 per leaf), everything else
   extension int32; the tree combines adjacent pairs; level `k` is reduced by a degree-3 sumcheck over `k`
   variables (`Σ_x eq(ρ,x)[p0 q1 + p1 q0 + λ q0 q1]`), the four child values at `x*` are sent, `μ, λ'` drawn.
   Sequential depth `1 + Σ_{k<n} (k+2)`.
4. **GKR per segment**, layer by layer: phase 1 = degree-3 sumcheck over the `b` copy variables of
   `Σ_c eq(γ,c) Σ_g w(g) Ã_g(c) B̃_g(c)`; phase 2a/2b = two degree-2 sumchecks over the `s_in` wire variables with
   the prover's `Ṽ(x) = Σ_c eq(c*,c) V_in(x,c)`; two claims on `V_in` merged with `α` into the next layer's `w`.
   Phase 1 is the batch-parallel work: tables `a[c,g], bb[c,g]` `(2^b, g_n)` base in round 0 (the `mm_mod` with the
   wiring), extension after the first fold; each round evaluates `h(t), t = 0..3` chunk by chunk
   (`a_t · b_t` elementwise extension products, then `Σ_g w_g (·)` as an exact float64 limb matmul, then the
   `eq` weight and the sum).  The `eq` tables are built by doubling (`eq_table`), folded with the same challenge.
5. **Residual functional** `⟨a, X⟩ = b`: the two input claims per segment (`eq(l*,·) ⊗ eq(c*,·)`), the LogUp leaf
   claim per table (every query's linear forms times `ρ λ eq(ρ_leaf, ·)`, one `mm_mod` per unit chunk against the
   query coefficient matrix), the chain (geometric coefficients), batched with fresh `ρ_1..ρ_m, ρ_geo`.
6. **Ligero opening** (`k = 4096, n = 16384, t = 192`, the CPU crate's parameters): `w = Σ r^i X_i` (proximity row),
   the `2k-1` coefficients of `q(Z) = Σ_i a_i(Z) U_i(Z)`, 192 distinct columns with SHA-256 paths.  Radix-2 NTT over
   the `2^27`-adic BabyBear roots (`ligero.ntt`, row-batched on the device); Merkle leaves/nodes by
   `backends/shared/hash_gpu` (SHA-256 CUDA), leaf = the column's u32 LE, node = SHA-256(left||right).

## 4. Transcript

SHA-256 Fiat-Shamir exactly as `../src/transcript.rs` (`state <- H(state || msg)`, challenges `H(state || "chal" ||
ctr)` read as LE u32 words with rejection sampling below `p`, six words per extension challenge); the CPU
protocol is Fiat-Shamir as well, so the diagnostic label is unchanged (`NON_ZK_PROOF_DIAGNOSTIC`).  Domain
`verity-gkr/candidate-a/v1/babybear6`.  **The GPU proof is verified by the ported Python verifier**
(`prover.verify`, ~700 lines, mirroring `main.rs/gkr.rs/logup.rs/ligero.rs` check by check); the Rust verifier
works over Goldilocks and cannot consume a BabyBear^6 proof.  Equivalence evidence: the tiny CPU run of both
implementations on the synth instances has identical message counts per phase and the same `rounds.protocol`
structure (the Goldilocks vs BabyBear circuits differ in size, so the counts are not equal — the check is
structural), plus the negatives.

## 5. Kernel plan

* v1 (this commit): everything as torch ops (int64 elementwise, `%` reductions, float64 limb matmuls for every
  contraction; `hash_gpu` for SHA-256).  Goal: a correct end-to-end proof of the full VU on the H100.
* v2: replace the phase-1 round kernel (the `a_t·b_t` products + `Σ_g w_g` contraction + `eq` weight, 4 evaluation
  points, both tables) with one fused Triton kernel over BabyBear^6 in registers (the `a-fused` lane's SIMT
  variant or a local one; `../tensor/reference.py` is the bit-exact reference), and the LogUp `round_poly` with the
  same kernel shape; store the folded tables as int32.
* v3: fuse the fold into the round kernel (read once per round), profile, attack the top bucket.

## 6. Memory (B = 4096)

Wires 1.2 GB (int32) + layer input rows 4.3 GB (int64, padded to 2^19 x 2^10) + phase-1 tables `a, bb` 4.3 GB each
(base) then 12.9 GB each after the first fold (int64 extension; int32 storage halves this) — peak about 45 GB in
v1; LogUp R16 levels ~20 GB (int32) freed level by level; Ligero code `(43.5k, 16384)` int32 2.8 GB, functional
`a` `(1.78e8, 6)` int64 8.5 GB.  Fits the 80 GB H100 without recomputation.
