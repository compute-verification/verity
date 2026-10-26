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
