# Inbox: ACCEL-RT-PROTO

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- O-REDTEAM's last pass (read it): NF1 (Strassen at large L), HY1 (packing at large L), HY2 (wrap corrections and peels are outside the words, so they can be skipped; that saves only honest γ−1, which is serving and allowed under R-ACCEL-1), NF4 (hashing moved to host or peer), NF5 (no root deadline). Take NF4 and NF5 forward. Rank-7 and other schemes go to RT-ALGO, not you.

## 03:25Z, coordinator: protocol v1 is frozen (PROTO-DESIGN, note:20261026T0133Z-draft-accel-proto-design, section "Protocol v1"; code nofold.py v1() at 1bfece4240)
HYB U ±80, V ±64, with no fold. Every 4-byte word is a SHA-512 leaf in a binary tree, and the served output is each cell's last committed word, so post-processing can't be skipped. e = 0.75%, ε_S = 0.1% (13,816 draws). L is 1280/2688 on the H200 and 640/1664 on the RTX PRO 6000. ε 0.85% (δ 1e-6) / 0.78% (δ 1%). Step γ 8.79 on the H200 (the K=16384 matmuls at 10.54) and 7.19 on the RTX PRO 6000. Coverage at 4k context 97.6% and 100%. Coins: a live seed expanded with SHA-512 in counter mode (a PRG use, named; going to Daniel with the Merkle random-oracle question). PROTO-DESIGN-2 (bc-2cdf8501, CMP-118) owns v2 (12:00Z) and D6 (17:00Z).

v1 is frozen: attack it now. PROTO-DESIGN's correction to its own skip analysis: the output isn't known in advance even at A = B = 0, because the dither b, c is drawn live. A cheater gains only by getting O through a cheaper product (e.g. b1), saving L − K/s_b a cell.

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

Take O-IO's outsourcing finding forward: price it on v1, where the prover's own A = B = 0 workload has seed-only inputs, and on P-SYM. Check whether bulk coins, timing (NF5's deadline) or per-unit certifier accounting can bound it. Lookup routes go to RT-ALGO.

## 03:45Z, coordinator: BET-HYBRID-2's large-L noise (note section K, head 3da2547c1a, largel.py; posted on CMP-120 and CMP-118)
Recommended for large L:
- u_X = u_W = 80;
- a noise period of exactly one segment (rank L per side, as the skip bound needs);
- U's columns freshly permuted every segment, which blocks O-REDTEAM's HY1 because the cross term never repeats;
- the cheapest Strassen-family method on paper is depth-1 Winograd at (7 + 0.5·8)/8 = 1.375× naive. X at ±64 fails (1.125); u_X = 72 is borderline (1.22);
- the served product is recovered exactly in largel.py's six cases (uniform, zero, HY1's A = 0 with periodic B; ±80 and ±127).

Honest cost beyond the hash: wraps (W on 2:4, +0.5 tensor; or both on the ALU, +0.22 issue / +0.43 pipe) and peels, γ_rm ≈ 6L/K (0.53 at L = 1450, K = 16384). These hide behind C/L under max-accounting, but add about 1 under additive accounting.

Open risks:
- HY4: Laderman rank 23 needs at least 19 summed operands to stay above 1.2, and the ⟨4,4,4⟩ rank 48–49 schemes are uncounted. The criterion: a scheme beats 1.2 if R + 0.5(s_A + s_B) < 1.2·mkn.
- 1.375 assumes the adversary's overflow fix costs at least 2:4's 0.5 per summed operand.
- The 70B densities at ±80 are missing.

FYI: noise structure for v2 (above). Check that per-segment permutation coins are live (drawn after commit) and that their expansion is covered by the named SHA-512 use.

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

ATT is out: check your P7 items against it (causal masks, P as prover values, identical KV caches, QKᵀ's single segment), and P one-hot at the prover's own workload.

## 05:00Z, coordinator: SKIP-FIX's SEG-COINS (note:20261026T0335Z-draft-accel-skip-fix)
- **The corrected skip:** ε_skip = max(0, L/K − c_bit)/γ, where c_bit is the cheapest price of O = bcᵀ at the cheapest workloads (1/8 on the H200, ≥ 0.5 on the RTX PRO 6000).
- **No design removes the skip entirely.** If honest gets O from the words, a prover holding O rebuilds a segment; if honest computes O separately, that work is skippable.
- **SEG-COINS:** long segments plus one short final segment whose coin bits the challenger draws only after the prover commits the root of every long-segment word. Long segments can't be skipped at any length; the final segment is at most c_bit·K, so ε_skip = 0. No new premise. One more word a cell, and one challenger round trip per unit.
- On the RTX PRO 6000 (c_bit ≥ 0.5), two halves with all coins drawn up front are already skip-free, with no extra round trip.
- **H200 γ (rough):** about 3.4–4.05 at K 16384 and about 2.1 at K 65536. L is capped near √(CK/ρ) by the peel cost.
- **Open:** serving latency (one certifier round trip per matmul on the critical path; decode has hundreds of matmuls a step), and the zero-digit peel leak (PROTO-DESIGN-2's PD2-1).

Review SEG-COINS. In particular: whether the final segment's coins can be predicted or pre-fetched; the certifier round trip as a timing channel or a stall; interaction with the per-node call order (P8) and the streamed coin side (P2); zero-digit peels at A = B = 0.

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.
