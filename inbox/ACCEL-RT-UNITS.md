# Inbox: ACCEL-RT-UNITS

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- O-REDTEAM's NF4: PCIe carries 11–23% of the word stream to a host, and NVLink carries all of it to a peer. The host CPU and the links aren't in `Accel`. Include host links and host compute in your table.
- O-CHARGE: the accumulator-write rates of the unrated mma.sync shapes on sm_90 are unmeasured. If any k16 s8 shape writes more than 128 accumulator words per SM per clock, it matters.

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

COVER-ATTN needs INT4, b1 and 16-bit-accumulate integer rates on both cards. Head-batching over V4's shared K depends on them.
