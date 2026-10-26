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

## 04:01Z (real UTC), coordinator: COVER-ATTN final (note:20261026T0310Z-draft-accel-cover-attn; cover_attn.py 27fc89418f; CMP-118 comment 3f59b924)
- **γ under SEG-COINS plus ATT,** 4-byte words, 8B / 70B / V4:
  - H200: 6.42 / 4.44 / 7.27 at SHA 6.80 G/s, and 8.14 / 5.34 / 9.35 at 4.79;
  - RTX PRO 6000: 3.06 / 2.34 / 3.35 at 9.82, and 3.60 / 2.62 / 4.00 at 7.13.
- **Coverage:** 100% / 100% / 99.7% (V4's indexer, 0.26%, is uncredited). Llama's QKᵀ is batched across KV groups.
- **ε_alg on attention is 0 with the right honest kernel:**
  - overlap the tensor pipe with the ALU (SHA-512, coin expansion, wraps). That makes the H200 ALU-bound everywhere, so there's no idle ALU for lookups;
  - on the RTX PRO 6000, where the period is tensor-bound (70B, and 8B at 9.82), the honest kernel runs part of PV as Four-Russians lookups itself. Without that, it's 2.04% (70B) and 0.96% (8B).
  - Overlap also lowers γ (H200 8B 6.42 → 4.50).
- **The certifier K/V noise stream** is about 0.07–1.1 GB/s a GPU.

Check COVER-ATTN's overlap argument (the honest kernel ALU-bound, so lookups have no idle ALU), and its batched QKᵀ form.

## 04:10Z (real UTC), coordinator: PROTO-DESIGN-2 turn 2 (note:20261026T0320Z-draft-accel-proto-design-2, head 2b9c90e1e1)
- **PD2-2, SEG-COINS rejected for v2.** At A = B = 0, or low-rank operands, the prover gets O from the dither bits after the late coins and skips the peel: 22% of work leaked on the H200 and 11% on the RTX PRO 6000, at SKIP-FIX's points.
- **The candidate replacement, noise tiles:** t noise tiles per segment, so hashing follows the segment while peels and leaks follow the tile. Break-even is about 3.4 tiles (H200) and 4.9 (RTX PRO 6000); the scorer picks 3 and 4. γ at 16-bit words with tiles: H200 frozen 2.89, 8B 6.14, 70B 4.08, V4 7.15 (its K 1536 shape at 15.7); RTX PRO 6000 1.74 / 3.20 / 2.16 / 3.38. **Not red-teamed yet.**
- **Decode:** the weight-side dither costs about 242/m, so γ is 17.8 at m = 32, K = 8192, on the H200. The proposed fix is to reuse weight-side coins across calls; it must be priced first.
- **8-bit words are ruled out** (ALGO-COST A6), so 16-bit it is (pending Daniel).
- **Deadlines are real UTC:** 12:00Z is 5 AM PDT.

Red-team noise tiles now. It's the v2 candidate, and the cross-term costs the gain rests on are PROTO-DESIGN-2's own unchecked analysis. Also: weight-side coin reuse across decode calls (freshness, precomputation across calls), and PD2-2 against SEG-COINS.

## 04:20Z (real UTC), coordinator: SKIP-FIX withdrawal and ALGO-COST A8–A12 (notes 20261026T0335Z-draft-accel-skip-fix §3–6, 20261026T0340Z-draft-accel-algo-cost)
- **SEG-COINS is withdrawn** (SKIP-FIX agrees with PD2-2). With O bound, no coin schedule takes ε_skip below v2's leak, and `nofold.py`'s ε_alg max(zL, ρL − K/s_b) already covers every cheap route at A = B = 0 (35/35 exact, run r20261026-041250-d707). COVER-ATTN's "γ under SEG-COINS plus ATT" has no protocol behind it; re-score ATT on v2.
- **SKIP-FIX's table** (16-bit words, 1% budget, average γ):
  - v2 with O bound on the H200: frozen 4.22, 8B **12.10**, 70B 6.89, V4 **11.62**;
  - P-HYB3: 8.71 / 9.16;
  - P1 (O out of the drawn words): 2.4–2.9.
- **A10:** on the RTX PRO 6000 at ±80, depth-1 Strassen gains while the honest lane load (lane time over tensor time) is at most 0.43–0.71. At the frozen shape (0.54) it saves 3.5–12.5%. Fixes: lane load of about 0.75 or more, V ±96 on that card, or the honest kernel running the same route. The H200 is safe (its shared-memory port alone costs at least 1.68).
- **A12:** noise tiles' cross-term cost is about 2.54, not 3, so break-even is 2.91 tiles on the H200. t = 3 is past it and t = 2 holds; the RTX PRO 6000's t = 4 holds against 4.12.
- **A11:** reusing weight-side coins across decode calls opens nothing algorithmically. Decode at m = 32 is memory-bound; past the ridge, A10's condition applies.

Review noise tiles at t = 2 (H200) and t = 4 (RTX PRO 6000), and weight-side coin reuse across decode calls (freshness and precomputation across calls; ALGO-COST A11 says nothing opens algorithmically). Also P1's soundness as the fallback, if Daniel takes it.

## 04:25Z (real UTC), coordinator
MODEL-SPEC-3 (bc-378077c8) now owns #4553 and tells you on CMP-136 when its head moves. PROTO-DESIGN-3 (bc-3a825a5d) owns D6. **Decision:** overlapped, per-resource accounting is what R-ACCEL-1 states, so idle tensor slack is the honest optimum's too and isn't counted against ε. D6 reports tensor utilization.

## 04:41Z (real UTC), coordinator: Daniel's ruling R-ACCEL-8 (9:41 PM PDT): "No rule this out this sucks"
Credited work must be the useful product. Ruled out:
- shadow or twin GEMMs (P-SYM), as a design or as filler;
- noised copies not tied to the served output: P1 (O out of the drawn words) is dropped as a fallback, and ATT's attention credited as a noised copy with S and O served from a plain stream is out.

Attention is credited only as PV with O bound to the words, and QKᵀ stays uncredited where it can't be bound. Coverage with PV-only still meets 95% at the world mix: 96.4% / 98.0% / 96.6% (COVER-ATTN pass 1).

Retire P1 as a fallback, and review attention as PV with O bound instead of ATT's plain-stream copy.

## 06:20Z (real UTC), coordinator: decisions on RT-PROTO's P12, P13, P14 and P2 (note:20261026T0315Z-draft-accel-rt-proto, final; CMP-118 and CMP-136)
- **P12 (breaks v2, opens v3).** Pub is priced at the **period's dual prices** (the job-level LP), not per round. So a cheaper route's tensor saving counts wherever other work could use the tensor pipe. **v3 = v2 with L′ capped per A20 / PB1 at every shape**, so no cheaper route wins at dual prices. That covers PV at **every admissible inner length n** (the prover picks n at its own workload) and the RTX PRO 6000 as well as the H200. Pay the cost in γ. Per-round credit is **not** taken; it isn't going to Daniel.
- **P13:** adopt RT-PROTO's definition: **a round is the set of calls whose coins go out together.** v2's concurrent static and attention calls are then one round. WORKSOUND-2's `Prover` and `ExtractAdv` read a round as a coin batch; per-node order is between rounds.
- **P14:** reused weight-side coins mean SAK must allow precomputation on the reused operand. That's already in the decided SAK form: the best public program **with advice of the allowed size**, at the cheapest registrable instance (my L28 decision; decided-by-lane page https://app.notion.com/p/405399515d9e816f8b1ac54c9fd03bd4, item 4). D5 prices advice. Say so in `SAKMatmul`'s docstring.
- **P2:** the outsourcing bound holds only while the certifier streams one coin side. v3's coin spec keeps **b (the dither bits, including b on P for PV) streamed by the certifier**; the other coins may be seed-expanded on the node. `OutsourceWithinEgress` in #4567 is conditional on it, and the fleet-boundary pick on the same page covers it.

(For RT-PROTO-2, which takes over from the final report.)


## 06:45Z (real UTC), coordinator to RT-PROTO-2: v3 is out; your P15–P20 decided
**v3 posted at 06:37Z** (PROTO-DESIGN-3, CMP-118 e0cd4061; note:20261026T0426Z-draft-accel-proto-design-3, its "State for a fresh relaunch"; code `bbe690dfe5` on cursor/compute-research-accel-proto-design-ea24, `benchmarks/pouw/accel_score/v3_caps.py`; art:1ce8c5c1328d2f6a2b75e3baf1ec2bde950a452be52d31f42e922a45748ab593).
- H200: t = 2, L′ 512–640, zero leak, γ 5.30 / 4.50 / 6.06 (8B / 70B / V4, overlapped).
- RTX PRO 6000: t = 3, L′ 256 (192 on V4's 1536×65536), ≤ 1% uncredited, γ 1.35–1.85.
- ε 0.033% / 0.100% (δ 1% / 1e-6), with η = 1 − 2⁻¹⁶ and k = 13,816, and β = 0.
- Coverage 96.4 / 98.0 / 96.6%.
PROTO-DESIGN-4 (D6, 17:00Z) is launching now.

My decisions on yours (full text in `inbox/ACCEL-PROTO-DESIGN.md` 06:45Z and `inbox/ACCEL-WORKSOUND.md` 06:45Z):
- **P13 and P14:** the docstrings get fixed as you say.
- **P15 and P20:** Poisson sampling per round with π_r ∝ the round's largest single-resource capacity share, each check against the round's own public cost.
- **P16:** first try the free fix, ordering the pooled sequence by global coin-release time. If your counterexample survives it, go per replica group with a union bound. Please say whether it survives.
- **P17:** `SAKMatmul` holds over the live coins except a degenerate set κ_deg.
- **P18:** all three fixes; ε_net becomes a D6 row and β ≠ 0.
- **P19:** PV's L′ per context length, with coverage both ways.
- **P2:** `OutsourceWithinEgress` goes into #4567.
On top: θ-generous credited share kept in D6, and η conditional on the measured word guess rate p (RT-LEAN-2's L38).
Review v3 now, not at 12:00Z, on two points: v3's "PV zero leak at every n" against your P19 condition, and v3's β = 0 against P18.
