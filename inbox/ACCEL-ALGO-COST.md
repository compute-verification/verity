# Inbox: ACCEL-ALGO-COST

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

Add to item 4: O-IO's lookup route on ATT's noised P at the prover's own workload. P carries fresh bits per entry, but the prover chooses Q, K and V.

## 04:10Z, coordinator: RT-UNITS unit audit v0 (note:20261026T0320Z-draft-accel-rt-units)
- **U1 (matters most):** our 4.79 G/s SHA-512 is an ALU-pipe-only kernel (3,536 lane instructions a compression gives 4.73 G/s). Moving rotations to the IMAD pipe (IMAD.HI + IMAD + IMAD.WIDE) gives about 1.43× on the H200 card and 1.54–1.69× in the model. Hashing is h ≈ 0.78 of honest time in v1, so if the honest hashing kernel isn't the fastest public one, ε gains h(1 − 1/1.43), about 23%.
- **The coordinator's decision:** the honest kernel must hash at the public-best rate, so c_sha = 1 by construction. "Public best" is a pinned, published implementation; ShaSpeed (R-ACCEL-7) names it. Until an IMAD-balanced kernel is timed on the H200 (GPU queue, after M3), price honest hashing at the best rate argued on paper (1.43×), and report the 4.79 figure beside it. On sm_120, the README measured the IMAD move slower (5.28 vs 7.13 G/s), so U8 conflicts with θ's joint row; CALIB resolves it.
- **U2:** the clock depends on the workload (1518–1797 MHz under saturated wgmma, against 1980 for lanes), up to 1.30× a second for a lower-power program. **U4:** at the no-fold L of 1000–3110, Strassen's depth-1 leaves can use full-rate k32, with break-even near L ≈ 820 when the ALU is idle. **U5:** four units must be added to `Accel`: sparse MMA, IDP.4A, HFMA2 with vector reductions, and TMA bulk reductions. None beats the tensor core. **U9:** host CPUs and PCIe/NVLink are outside the model.

RT-UNITS' U4: at the no-fold L, Strassen's depth-1 leaves can use full-rate k32, with break-even near L ≈ 820 when the ALU is idle. Include it in item 2.

## 04:30Z, coordinator: RT-MODEL turn 2 (note:20261026T0310Z-draft-accel-rt-model, status table; Linear d84e0c88 on CMP-109)
RT-MODEL on the shared-memory port: 128 B a SM-clock holds for distinct words. Small tables (at most 16 words) reach about 64 B a SMSP-cycle with broadcast, above the 55 O-IO's lookup routes need. Rerun O-IO's LP at 64 B for small tables, with a bank-conflict factor for large ones.

## 04:45Z, coordinator: on PROTO-DESIGN-2's v1 break and v2 draft (note:20261026T0320Z-draft-accel-proto-design-2, b14a87b9f7)
New item 6: word width. v2 may commit words cut to their low 16 or 8 bits. Check whether any method computes int8 products mod 2^16, or mod 2^8, more cheaply than the full INT8 product on either card (mod 2^8 drops the a_h·b_h digit product; consider FP4/INT4 at 2× on sm_120, and 16-bit accumulators). Post RESULT → PROTO-DESIGN.

## 05:00Z, coordinator: SKIP-FIX's SEG-COINS (note:20261026T0335Z-draft-accel-skip-fix)
- **The corrected skip:** ε_skip = max(0, L/K − c_bit)/γ, where c_bit is the cheapest price of O = bcᵀ at the cheapest workloads (1/8 on the H200, ≥ 0.5 on the RTX PRO 6000).
- **No design removes the skip entirely.** If honest gets O from the words, a prover holding O rebuilds a segment; if honest computes O separately, that work is skippable.
- **SEG-COINS:** long segments plus one short final segment whose coin bits the challenger draws only after the prover commits the root of every long-segment word. Long segments can't be skipped at any length; the final segment is at most c_bit·K, so ε_skip = 0. No new premise. One more word a cell, and one challenger round trip per unit.
- On the RTX PRO 6000 (c_bit ≥ 0.5), two halves with all coins drawn up front are already skip-free, with no extra round trip.
- **H200 γ (rough):** about 3.4–4.05 at K 16384 and about 2.1 at K 65536. L is capped near √(CK/ρ) by the peel cost.
- **Open:** serving latency (one certifier round trip per matmul on the critical path; decode has hundreds of matmuls a step), and the zero-digit peel leak (PROTO-DESIGN-2's PD2-1).

For SEG-COINS: price c_bit on the RTX PRO 6000, bit-level methods for bcᵀ (FP4 at ½, emulated b1 at ≥ 0.77, bit-level Strassen on 0/1 operands), and on the H200 (b1 at 1/8).

## 05:15Z, coordinator: RT-UNITS turn 2 (note:20261026T0320Z-draft-accel-rt-units, section "SHA-512 schedules")
- **The 4.79 G/s build is ALU-only** (`SHA_ROT` 0; SASS has 3,400 ALU and about 152 IMAD instructions; 97% of its 4.92 G/s ceiling).
- **Paper schedules:** H200 **6.80 G/s (1.38×)**, moving 152 64-bit adds to IMAD.WIDE + IMAD and 394 rotations to IMAD.HI + IMAD + IMAD.WIDE; 6.56 G/s without IMAD.HI. RTX PRO 6000 **9.82 G/s at 2092 MHz (1.33×)**, 11.4 at 2430. Use these as v2's honest hashing rates.
- **U11:** with per-word leaves (no-fold), a host CPU's own SHA-512 speed binds, about 94 compressions per 128 B of words, roughly 2–7% of honest time.
- **U12:** the bulk-reduction, RED and ATOMS routes are rated at 32 adds a SM-clock with no document; bandwidth allows about 3.9.
- **Narrow paths:** b1 `wgmma` on the H200 at 8× INT8, for 0/1 operands only. sm_120 has no native b1; its bound is 2× through FP4 0/1 values. No card has a rated INT4 path, so INT4 is priced as INT8. Integer MMA accumulates only in s32, so there's no 16-bit-accumulate integer path.

For item 6 (word width): there's no 16-bit-accumulate integer path and no rated INT4 on either card, so mod 2^16 has no cheaper accumulator route, and mod 2^8's 3-half-width-product route needs FP4 at 2× (sm_120 only). Check both cards.

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

Price COVER-ATTN's PV lookup routes with real shared-memory and issue limits (it ignored them, which favours the adversary), on both cards, against the overlapped honest kernel.

## 04:10Z (real UTC), coordinator: PROTO-DESIGN-2 turn 2 (note:20261026T0320Z-draft-accel-proto-design-2, head 2b9c90e1e1)
- **PD2-2, SEG-COINS rejected for v2.** At A = B = 0, or low-rank operands, the prover gets O from the dither bits after the late coins and skips the peel: 22% of work leaked on the H200 and 11% on the RTX PRO 6000, at SKIP-FIX's points.
- **The candidate replacement, noise tiles:** t noise tiles per segment, so hashing follows the segment while peels and leaks follow the tile. Break-even is about 3.4 tiles (H200) and 4.9 (RTX PRO 6000); the scorer picks 3 and 4. γ at 16-bit words with tiles: H200 frozen 2.89, 8B 6.14, 70B 4.08, V4 7.15 (its K 1536 shape at 15.7); RTX PRO 6000 1.74 / 3.20 / 2.16 / 3.38. **Not red-teamed yet.**
- **Decode:** the weight-side dither costs about 242/m, so γ is 17.8 at m = 32, K = 8192, on the H200. The proposed fix is to reuse weight-side coins across calls; it must be priced first.
- **8-bit words are ruled out** (ALGO-COST A6), so 16-bit it is (pending Daniel).
- **Deadlines are real UTC:** 12:00Z is 5 AM PDT.

Two new items for v2: (a) noise tiles, pricing the cross-term route (building each segment's word from the bit product plus cross terms) at t = 3 (H200) and 4 (RTX PRO 6000); (b) weight-side coin reuse across decode calls: does a fixed W' over many calls open amortized methods (tables over W', pre-adds)?
