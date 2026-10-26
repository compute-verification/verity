# Inbox: ACCEL-COVER-ATTN

## 03:25Z, coordinator: protocol v1 is frozen (PROTO-DESIGN, note:20261026T0133Z-draft-accel-proto-design, section "Protocol v1"; code nofold.py v1() at 1bfece4240)
HYB U ±80, V ±64, with no fold. Every 4-byte word is a SHA-512 leaf in a binary tree, and the served output is each cell's last committed word, so post-processing can't be skipped. e = 0.75%, ε_S = 0.1% (13,816 draws). L is 1280/2688 on the H200 and 640/1664 on the RTX PRO 6000. ε 0.85% (δ 1e-6) / 0.78% (δ 1%). Step γ 8.79 on the H200 (the K=16384 matmuls at 10.54) and 7.19 on the RTX PRO 6000. Coverage at 4k context 97.6% and 100%. Coins: a live seed expanded with SHA-512 in counter mode (a PRG use, named; going to Daniel with the Merkle random-oracle question). PROTO-DESIGN-2 (bc-2cdf8501, CMP-118) owns v2 (12:00Z) and D6 (17:00Z).

v1 is frozen. On the H200, QKᵀ is at γ 176 and not credited, PV is credited at long context, and coverage at 4k/8k/32k is 97.6%/95.5%/85.7%. On the RTX PRO 6000 PV and QKᵀ are credited: 100%/100%/85.7%. Hand your design to PROTO-DESIGN-2.

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

FYI: under P-SYM, attention's MACs are credited by twins of any shape (large K), so coverage reaches 100% cheaply. Keep designing for the useful-credit line, and note in your table what P-SYM would give.

## 04:50Z, coordinator: RT-PROTO's P2/P9 fix and ALGO-COST's first pass
RT-PROTO's P10: attention units can be outsourced under P2 (6–9% of credited time). The fix is one streamed coin side per form; decode PV is up to 10.7 GB/s a GPU. Its P7 (iii): specify decode noise lifetime (decode γ about 1.7).

## 05:15Z, coordinator: RT-UNITS turn 2 (note:20261026T0320Z-draft-accel-rt-units, section "SHA-512 schedules")
- **The 4.79 G/s build is ALU-only** (`SHA_ROT` 0; SASS has 3,400 ALU and about 152 IMAD instructions; 97% of its 4.92 G/s ceiling).
- **Paper schedules:** H200 **6.80 G/s (1.38×)**, moving 152 64-bit adds to IMAD.WIDE + IMAD and 394 rotations to IMAD.HI + IMAD + IMAD.WIDE; 6.56 G/s without IMAD.HI. RTX PRO 6000 **9.82 G/s at 2092 MHz (1.33×)**, 11.4 at 2430. Use these as v2's honest hashing rates.
- **U11:** with per-word leaves (no-fold), a host CPU's own SHA-512 speed binds, about 94 compressions per 128 B of words, roughly 2–7% of honest time.
- **U12:** the bulk-reduction, RED and ATOMS routes are rated at 32 adds a SM-clock with no document; bandwidth allows about 3.9.
- **Narrow paths:** b1 `wgmma` on the H200 at 8× INT8, for 0/1 operands only. sm_120 has no native b1; its bound is 2× through FP4 0/1 values. No card has a rated INT4 path, so INT4 is priced as INT8. Integer MMA accumulates only in s32, so there's no 16-bit-accumulate integer path.

Your ask, answered: no rated INT4 path on either card (price as INT8); no 16-bit-accumulate integer MMA; b1 at 8× on the H200 for 0/1 only; sm_120 FP4 0/1 at 2×.

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.

## 04:41Z (real UTC), coordinator: Daniel's ruling R-ACCEL-8 (9:41 PM PDT): "No rule this out this sucks"
Credited work must be the useful product. Ruled out:
- shadow or twin GEMMs (P-SYM), as a design or as filler;
- noised copies not tied to the served output: P1 (O out of the drawn words) is dropped as a fallback, and ATT's attention credited as a noised copy with S and O served from a plain stream is out.

Attention is credited only as PV with O bound to the words, and QKᵀ stays uncredited where it can't be bound. Coverage with PV-only still meets 95% at the world mix: 96.4% / 98.0% / 96.6% (COVER-ATTN pass 1).

ATT's plain-stream noised copy is ruled out. Attention = PV with O bound (your pass-1 design); PROTO-DESIGN-3 carries it into v2.
