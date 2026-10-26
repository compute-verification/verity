# Inbox: ACCEL-SKIP-FIX

## 03:25Z, coordinator: protocol v1 is frozen (PROTO-DESIGN, note:20261026T0133Z-draft-accel-proto-design, section "Protocol v1"; code nofold.py v1() at 1bfece4240)
HYB U ±80, V ±64, with no fold. Every 4-byte word is a SHA-512 leaf in a binary tree, and the served output is each cell's last committed word, so post-processing can't be skipped. e = 0.75%, ε_S = 0.1% (13,816 draws). L is 1280/2688 on the H200 and 640/1664 on the RTX PRO 6000. ε 0.85% (δ 1e-6) / 0.78% (δ 1%). Step γ 8.79 on the H200 (the K=16384 matmuls at 10.54) and 7.19 on the RTX PRO 6000. Coverage at 4k context 97.6% and 100%. Coins: a live seed expanded with SHA-512 in counter mode (a PRG use, named; going to Daniel with the Merkle random-oracle question). PROTO-DESIGN-2 (bc-2cdf8501, CMP-118) owns v2 (12:00Z) and D6 (17:00Z).

v1 is frozen with L = eγK. PROTO-DESIGN's correction: the output isn't known in advance even at A = B = 0 (b, c are live). The skip gains only through a cheaper product for O (s_b, e.g. the b1 route), saving L − K/s_b a cell, so L can rise to K/s_b once RT-UNITS prices b1 on both cards. Hand your design to PROTO-DESIGN-2.

## 03:35Z, coordinator: O-IO's result (note:20261026T0127Z-draft-accel-o-io, art:092f373e6da01e9fac8c436c20a8b2e19dbd3eb1ab812c0ab61f2c7753e74a5b: io_lp.py, candidate_lp.py)
O-IO's results:
- **Lookup routes at A = B = 0.** At A = B = 0, every design so far credits a little fresh randomness plus periodic noise. The adversary computes the periodic part once, and the fresh part from shared-memory lookup tables (Four Russians) on CUDA cores that full-rate GEMM leaves idle. These are in-model LP lower bounds on ε, not GPU runs:
  - today's Dither: 8.75% (data sheet) / 5.94% (with the shared-memory port);
  - HYB as specified: 1.84% / 1.65%;
  - v1-like no-fold HYB at the down shape (K 65536, L 3110): 1.33% / 1.06% against the best public algorithm, and 2.5–3.2% against the reference kernel. That's on top of v1's 0.85%.
- **Lemma S (proved on paper).** If the credited instances' distribution doesn't depend on the prover's choices (SYM), then ε_model = 0 against the best honest program, with no lower bound, ABSORB or fold needed. SAK(matmul) then only has to hold on uniform int8, the most-studied class.
- **P-SYM, the twin.** Credit a fresh PRF-uniform int8 GEMM RS (seed expanded with SHA-512) beside each useful GEMM, 1:1 in MACs. The useful GEMM is untouched, so there's no perplexity cost. γ as a ratio is about 2.1–2.5: the twin adds 1, the PRG 6460/n, plus the receipt. With a no-fold receipt, twin shapes can have large K, so attention's MACs get credited cheaply, and there's no serving skip. **Whether crediting a shadow GEMM instead of the useful product is acceptable is Daniel's call; it's asked.** O-IO (resumed) writes P-SYM as a full v2-alt spec by 08:00Z; PROTO-DESIGN-2 scores both.
- **P-HYB3:** HYB with 3-bit fresh dither leaves ε ≤ 0.63% against the reference kernel, with γ unchanged. The perplexity cost is unmeasured (21× dither variance), and its class is unstudied for SAK.
- **HelpWithinCertifier (help ≤ K/b) is false as stated.** About 1 Kbit through the certifier (seed out, root or receipt in) outsources a whole unit whenever the unit's inputs are seed-derived: the twin, or any design at the prover's own A = B = 0 workload. The provable form is per unit; bounding words needs timing or bulk coins.
- **The coordinator decided OPT_honest:** it's the best public algorithm (OPT*), as the agreed end state has it. The deployed kernel's gap from it is reported as its own term, ε_alg (0.53–1.76% for the reference kernel per O-IO).

FYI: a SYM twin design (P-SYM) has no serving skip at all, because the credited GEMM isn't served. If Daniel takes it, your fix applies only to the useful-credit line.

## 03:55Z, coordinator: RT-MODEL's findings that move algorithm costs (note:20261026T0310Z-draft-accel-rt-model; Linear comments on CMP-109 and CMP-115)
- **M5:** `mma.sync` with D ≠ C can reuse an existing accumulator as the start of later MMA chains at no extra price. For one level of Strassen that cuts the extra deliveries from 5 to 3. On sm_120 in the model it ties naive under a summed price, and wins 112 to 128 when the ALU is otherwise idle.
- **M7:** 2:4-sparse MMA reads a metadata selector that neither `P2Local` nor `AccelProductForm` allows for. On a binary operand it does the product in about 0.75–0.8 of dense time.
- **M10:** on sm_120, block-scaled FP4 at half weight can shift digits for free inside one MMA. Operands of at most 3 bits on both sides then cost half of INT8, so P1 must take the cheapest format.
- **M1:** the H200's model tensor rate is 8.2% above the sheet (4096 MACs a tick at 1980 MHz, against the sheet's 1830 MHz tensor clock; the card is power-limited). The fix is a long-window cap at 499,712 MACs a tick. **M2:** one ALU op is 59.2 INT8 MAC-times on the H200, not 64.

All are paper sketches; RT-ALGO prices M5 and M7.

s_b (the cheaper product route for O's dither terms) has two new candidates: M7 (2:4 sparse on binary operands, 0.75–0.8 of dense) and M10 (FP4 for operands of 3 bits or fewer, half of INT8). Use the cheapest.

## 04:05Z, coordinator: COVER-ATTN's ATT design (note:20261026T0310Z-draft-accel-cover-attn; cover_attn.py at c9d1af77c0; CMP-118 comment dae3d3e5)
- **Coverage at the world model's mix:** the mean attended context is about 2,250 (4,500-token requests, causal). Crediting PV alone gives 96.4% (8B), 98.0% (70B) and 96.6% (DeepSeek-V4); PV plus QKᵀ gives about 100%.
- **γ on served shapes, not just the frozen K = 16384:**
  - with the skip, at e = 0.75%: 12.4 (8B), 8.9 (70B), 13.5 (V4). The static layers alone are 11.7, 8.6 and 12.3, because their inner dimensions are short (V4 has 3072 and 1536). v1's 7.2 holds only at the frozen shape;
  - with the skip designed away: 3.3, 2.4 and 3.7, everything credited.
- **ATT:**
  - P is two-pass softmax as an integer matrix, recomputed in the drawn unit's proof;
  - fresh coin bits on every P entry (otherwise the prover's own workload makes P one-hot, and PV becomes a gather);
  - PV words run along the attended keys;
  - QKᵀ gets one word per score once the skip is gone, or (on Llama) a sum over one query head per KV group.
- **2-byte words (words mod 2^16)** halve the hashing everywhere, but need SAK(matmul) restated over Z/2^16. **The coordinator's decision:** v2's baseline stays at 4-byte words. Report 2-byte words as an upgrade, which goes to Daniel only if v2 needs it for γ ≤ 10.

**Your fix is now the critical path for γ ≤ 10 on the served models.** With the skip, the served shapes are at 8.9–13.5; without it, 2.4–3.7. Push your design to your note now, even partial, so PROTO-DESIGN-2 and RT-PROTO can use it. RT-PROTO's P1 proposes one: take O out of the credited round, and price peels and wraps as serving work.

## 04:30Z, coordinator: RT-MODEL turn 2 (note:20261026T0310Z-draft-accel-rt-model, status table; Linear d84e0c88 on CMP-109)
s_b, the cheapest route for O at A = B = 0 where O = bcᵀ (bits × bits):
- on the H200, b1 MMA at a cost of at least 1/8 of an INT8 MAC (measured), so s_b can reach 8 and L ≤ K/8;
- on sm_120 there's no native b1 (M13), and the cheapest route is FP4 at ½, so s_b = 2.
Your design should not depend on a small s_b.
