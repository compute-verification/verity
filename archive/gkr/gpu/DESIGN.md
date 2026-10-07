# A-GKR on the GPU: design

Track A's prover (GKR + LogUp + Ligero, `../PROTOCOL.md`) in torch (+Triton) over BabyBear^6, for the full
K=1536 VU (96 chained units + epilogue, PROTOCOL §12) at batch 4096 on one H100.  The CPU Rust crate
(`../src/`) is the reference for the *protocol*; this port keeps its message order, layer order, LogUp tree,
Ligero parameters and Fiat-Shamir structure, and changes the field.

## 1. Field, circuit, embedding

* Base field `p = 2^31 - 2^27 + 1` (BabyBear), challenges in `F_{p^6} = F_p[X]/(X^6 - 31)` (the same `W = 31` as
  `../tensor/reference.py`).  Tables are u32 BabyBear (int32 tensors, int64 compute); an extension element is six
  u32 in planar order `c0..c5` (a trailing dimension of 6).
* The circuit is the WP2 checker **under the red-team's BabyBear embedding** (`note:r20-proof/red-team-babybear/20260922T0605Z-report-babybear-rules`): it is
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
   `verity/ml/kernels/hash_gpu` (SHA-256 CUDA), leaf = the column's u32 LE, node = SHA-256(left||right).

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

* v1: everything as torch ops (int64 elementwise, `%` reductions, float64 limb matmuls for every
  contraction; `hash_gpu` for SHA-256).  A correct end-to-end proof of the full VU on the H100: 26.95 s.
* v2 (done, `kernels.py`, each step bit-exact against the torch path and on the ledger): fused Triton SIMT
  BabyBear^6 kernels in 32-bit Montgomery form for `ext_mul`, the fold, the LogUp degree-3 round, the GKR phase-1
  round (2D tiles over copies x gates), and the lookup functional as a CSR scatter over the sparse (query, column)
  terms; the Ligero opening as modular matmuls on tensor cores (`G = Mᵀ (aᵀ X) M`).  8.3 s at v2c.  The torch path
  stays as the reference (`VERITY_GPU_TORCH_ONLY=1`).
* v3 (next): sparse gate evaluation (`Layer.eval_gates`), int32 storage of the folded tables, the fold fused into
  the round kernels (one read per round), a batched NTT for the commitment, a compiled verifier.

## 5b. Checker v2 in the GPU prover (`--variant v2`, lane a-gpu-v2, 2026-09-22; `gpu/v2/`, PROTOCOL.md §14)

The v2 circuit of `gkr_export.v2` (264 columns / 290 wires / 26 products / 86 lookups per unit into 10 tables,
depth 1; epilogue 16 columns; the `(s, e, M, z)` state chained across the 96 units, 583 linkage constraints per VU)
exported over BabyBear by `gpu.v2.export` — the exporter is field-generic, only its `FieldApi` refused BabyBear —
and proved by the *same* prover, verifier and kernels as v1.  What v2 needed in the shared code is additive:

* **Implicit tables on device** (`circuit.Table.implicit`, `circuit.materialise`): `itable NAME cols rows shift|tnorm
  params` lines are parsed into a `Table` without rows; `device_rows` materialises `(rows, 2)` int64 on the device
  on first use (`SHIFT` 3.54M rows: one integer divide, one shift, one select), cached per device.  `Table.nrows`
  is the row count for row-listed, range and implicit tables alike; `dense` (key = row index) makes the
  multiplicity pass a bounds check + gather-compare + `bincount` instead of argsort + searchsorted.
* **Range tables from the queries** (`circuit.rs` rule): every `R<w>` a circuit's queries reference, narrowest
  first, as an implicit `arange` table — v1 circuits reference `R16` only, so their table list is unchanged; v2's
  unit has `R4`, `R12`, its epilogue `R15`, `R16`.
* **Table union over segments** (`prover.union_tables`): the instance's tables are segment 0's in order, then every
  later segment's new ones.  `main.rs::Instance::tables` returns segment 0's list only, so in the Rust a-v2 prover
  the epilogue's 4 `R16` + 1 `R15` queries are never part of any LogUp argument (v1's epilogue only used the unit's
  `R16`, so v1 is unaffected) — reported to the coordinator; the GPU path proves all 12 tables.
* **Chain links** (`prover.Link`, `Chain.links`, `add_chain_links`, `check_chain_links_clear`; `run.read_chain`
  parses `chain.txt`): the general `main.rs::Link` — `col_c` set: `X[(v,0),c] = init`, `X[(v,i),c] = X[(v,i-1),y]`,
  epilogue link; `col_c` None: epilogue link and, if `zero_nonlast`, `X[(v,i),y] = 0` for `i < steps-1`.  The
  functional follows `main.rs::add_chain` link by link (per VU: link 0's 96 + 1 constraints, ..., then the public
  word) as strided reads of the 12 linked columns; the v1 limb path is untouched (`links is None`).  The
  sha256 of `chain.txt` is absorbed in the preamble after the public words (the links are public data; the Rust
  crate binds them through the circuit hash only).
* **Two checker layers**: `layers()` already handles `max_depth = 1` (assertions `s_out = 5` over the 29
  assertions, then one layer `s_in = 9` reading the 264 columns), so the phase-1/2 sweeps drop from 5 to 2 per
  segment; `depth_circuit` 63 at B = 64 (v1: 197 at B = 4096).
* **12 lookup trees** instead of 5: `T_OP` (n = 24 at B = 4096), `SHIFT` (24), `ALIGN4`, `SSHIFT_HI/LO`, `LEAD`,
  `LEADNORM`, `TNORM`, `R4`, `R12`, `R15`, `R16`; the deepest is `n = 24` (v1's `R16` tree was `n = 28`), so the
  lookup chain and with it `rounds.sequential_depth` shorten (279 vs 438).  The 5.14M table-side leaves are a fixed
  per-proof cost (1.9 s of the 2.75 s at B = 64).

**Field caveat (the reason for the label).**  checker-min's z3 results (`note:r20-proof/checker-min/20260922T0727Z-report-checker-min` 3.2) prove the v2 gadgets
UNSAT over Z and Goldilocks; over 31-bit fields the state-chained VU body is UNSAT by wrap-free transfer (M31), but
`epilogue` (and the absent `state_decode`) are SAT as stated: `Pack.check` forms the FP32 word `s 2^31 + e 2^23 + M`
as one field element and `Epilogue.check` splits it as `hi 2^16 + lo` — over BabyBear (`p < 2^31`) a second
`(hi, lo)` pair satisfies the congruence, so a wrong output word can be certified.  The a-v2 export was therefore
Goldilocks-only.  Choice made here: **run the unchanged v2 circuit over BabyBear^6** (the field every kernel of
this prover is written for; a Goldilocks^3 port would rewrite `field.py`, `kernels.py`, the NTT and the Ligero
matmuls for 64-bit arithmetic — 4–8× the SIMT work per product with 32-bit-limb emulation — and was priced at more
than the night) and **repair the epilogue's embedding**, which is the only SAT row.  The first end-to-end numbers
(r20260922-090930-3bc5, 5.62 s) ran a-v2's word-form epilogue and are labelled `INCOMPLETE_CHECKER_DIAGNOSTIC`
naming those 16 columns; the export default is now the limb form and the second set carries the honest label.

**Limb-form epilogue (`gpu/v2/epilogue_limbs.py`).**  `lane/b-air-v2`'s `pack_limbs` / `epilogue_limbs`
(`V2_AIR.md` 4.5; `notes-asset:campaigns/r20-proof/assets/b-air-v2/reports/b_air_v2_z3.json` `pack_epilogue_limbs`: integer UNSAT, wrap-free UNSAT and *direct*
UNSAT with the BabyBear modulus at REAL widths), copied verbatim: `M = M_hi8 2^16 + M_lo16` with the two hints
range-checked (`R8`, `R16`), the finite high limb the *expression* `s 2^15 + fld 2^7 + M_hi8` (below `2^16` on every
admissible state, so no wrap is possible), the infinity blend per limb, then v1's epilogue on the limb pair
(PROTOCOL §13 LIMB16: `lsb_split`, `round`, `y16 = y_hi + carry`).  `y_hi 2^16 + y_lo` equals `Pack.word`
identically over Z (`gpu/v2/tests/test_epilogue_limbs.py`, 5000 admissible states), so the relation is unchanged —
same function, different embedding.  Epilogue: 19 columns / 26 wires / 9 asserts / 7 range queries (`R8` ×1, `R16`
×5, `R15` ×1) per VU vs 16 / 22 / 6 / 5 for the word form; the 9 leading columns (`one, s, e, M, z, ovf[g], sgn[g]`)
are the same, so `chain.txt` is byte-identical (`chain_sha256 0802…42a47`).  The `gkr_export.v2` synth and
negatives generators keep the word form (their hint paths are the word form's; `BabyBearV2Api("word")`); the
bench-instances export uses `"limbs"` and records `"epilogue": "limbs"` in the manifest.  `gpu.run
epilogue-negatives` bumps every base column of one VU's epilogue row (state, flags, the word, all six hints) by +1,
recomputes the derived wires and requires the GPU prover/verifier to reject — a bumped hint pair is exactly the
BabyBear wrap attack on the word form.

**Unit body over BabyBear.**  checker-min's z3 queries were run for M31 and Goldilocks only; re-running
`checker.v2.search` with the BabyBear modulus (`gpu/v2/results/z3_babybear_v2.json`, 120 s direct budget) gives
the M31 pattern exactly: `state_shift`, `signed_sum`, `normalise`, `pack` wrap-free UNSAT (so the integer UNSAT
transfers), `group_max`, `product_shift`, `signed_sum`, `op_decode`, `product`, `pack` direct UNSAT, `group_max`
wrap-free SAT (the harmless hint-`E` class, direct UNSAT), `state_decode` SAT (absent from the chained VU body),
`epilogue` SAT as stated (the row the limb form replaces).  With the limb epilogue every gadget the GPU prover
checks is therefore verified over the field it runs on, and the label is `NON_ZK_PROOF_DIAGNOSTIC`.

## 6. Memory (B = 4096)

Wires 1.2 GB (int32) + layer input rows 4.3 GB (int64, padded to 2^19 x 2^10) + phase-1 tables `a, bb` 4.3 GB each
(base) then 12.9 GB each after the first fold (int64 extension; int32 storage halves this) — peak about 45 GB in
v1; LogUp R16 levels ~20 GB (int32) freed level by level; Ligero code `(43.5k, 16384)` int32 2.8 GB, functional
`a` `(1.78e8, 6)` int64 8.5 GB.  Fits the 80 GB H100 without recomputation.
