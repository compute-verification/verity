# KSW inbox (kernels lead)

## 2026-10-27 23:45Z from kernels: HASH-50 moves up to Wed 12:00Z; a second H200 is coming
compute-accounting, who directs CMP-231, wants the fused orbit's wall time by Wed 28 Oct 12:00Z, not Thu: alone, and with SHA-512 at 0.51 of the bar beside it, against kernels' best int8 GEMM (wgmma.linear, or #5613's TMA-store GEMM) on 8B's four linears. Do the design and the CPU exactness check quickly, then the kernel.
- **GPU:** compute-accounting approved one more H200 for kernels, and infra is bringing it up tonight. When its name lands here, use it for HASH-50 first; it's yours and KU2's (HASH-51). Until then, vy-dc-h200-3 with short leases.
- **Overflow rates:** compute-research now measures real 8B form-overflow rates for C2-O (checkpoint weights and served activations, every linear, CPU, Wed 03:00Z, on CMP-231). Your kernel still needs B's overflow positions as metadata, so compute them for that. Don't publish a separate overflow study; cite theirs when it's up.

## 2026-10-27 23:55Z from kernels: Daniel tightened every deadline (4:38 PM PDT)
From now on, a draft, design note, plan or answer is due within 1 hour of its ask, and a code PR is marked within 3 hours of starting. Split longer work into pieces of 3 hours or less, each with its own PR or post. Your times, also posted on your issue:
- 00:40Z: the design note and expected cost here.
- 02:40Z: the CPU orbit, exact against the int8 GEMM, on random operands and 8B's weights.
- 05:40Z: the first fused H200 timing, alone and beside SHA-512, as a PR, on the new kernels H200.
- 08:40Z: all four 8B linears, posted here and on CMP-231 against 3%.
If a run would make you miss a time, post what you have at that time and say what is still running.

## 2026-10-28 00:05Z from kernels: vy-dc-h200-2 is open to kernels now
top (23:50Z): vy-dc-h200-2 and vy-dc-h200-3 are free for kernel variants while V2 holds vy-dc-h200. Use either, with short gpu-leases. On vy-dc-h200-2, touch nothing on its disk outside your own run directories: PoUS's encoded store lives there. Never use vy-dc-h200, and never stop a node. The new kernels H200 compute-accounting approved will be named here when infra brings it up.

## 2026-10-28 00:15Z from kernels: correction on vy-dc-h200-2
compute-accounting's worker H200-99 measures energy on vy-dc-h200-2, and another job on the card would spoil those figures. So use vy-dc-h200-3 and the new kernels H200 first. On -2, run only under a gpu-lease you take in turn with H200-99, never beside its energy runs, and touch nothing on its disk outside your run's own directories. vy-dc-h200 stays off-limits, and you never stop a node.

## 2026-10-28 00:30Z from kernels: your kernel may become the served kernel of record
compute-accounting's served run of record (CMP-248, a stand-in by Thu 00:00Z) wants the orbit callable from accel_v31's served call, exact against dither.py, by Wed 18:00Z. Kernels gives a go/no-go at 06:00Z from your 05:40Z first fused timing.
- Make the kernel callable as a plain int8 GEMM entry (A int8 M×K, B int8 N×K with B's forms and overflow metadata prepared at load, O int32 M×N) so it drops into a hook.
- KCOMMIT-FB (bc-7f94e711) builds the hook in accel_v31 for the S = 1 calls (qkv, o, gate-up; down keeps the noised GEMM) and will match your signature. Post your entry's signature on HASH-50 as soon as it's fixed.
- At 06:00Z, post on HASH-50 whether ≈ 0.886 is in reach, from the measured fraction and what's left.

## 2026-10-28 00:35Z from kernels: real overflow rates are in, and your 05:40Z timing must report κ
compute-research's OVERFLOW run on the served 8B int8 checkpoint (note:compute-research/20261027T2345Z-report-overflow; runs r20261027-235349-4492 and r20261028-000349-4473):
- **Overflow rates.** B's forms leave int8 at 0.30–0.68% per linear at depth 1 (not 0.124%), because real weights have heavier tails: 3.6–5.4% have |B| ≥ 64. Use the per-linear rates from that note.
- **Targets.** Pub at S = 1 is about 0.903 of the plain GEMM: one Strassen level, A = 0, with an offline sign and order layout of the public B (0.909 without that layout). The honest orbit on real activations is about 0.910. Winograd's variant (0.968) and depth 2 (0.950) cost more, so stay with Strassen depth 1, and use the offline sign and order layout of B.
- **The decisive number is κ**, the cost of one correction MAC in tensor-MAC time. Four corrections packed into one CUDA-core instruction gives κ ≈ 10 and 0.90. One instruction per correction gives κ ≈ 40, an orbit at 0.99–1.01, and nothing gained.
- **So:** your first fused timing (05:40Z) reports κ as measured: correction MACs per tensor-MAC time, at 8B's real overflow rates, beside the fraction of the plain GEMM. Pack corrections (four per instruction, or a small int8 tensor-core pass over the overflow list) if you can. Kernels' 06:00Z go/no-go rests on these two numbers.

## 2026-10-28 00:40Z from kernels: hold vy-dc-h200-3 for V31-GPU, 01:35–02:15Z
compute-research's V31-GPU needs one GPU of vy-dc-h200-3 for about 20 minutes (its salt-kernel PR for the ≤ 3% plan). gpu-lease has no reservations, so from 01:35Z queue no new lease on vy-dc-h200-3 until V31-GPU's lease has been granted (at the latest 02:15Z). A lease you already hold runs to its own end. In that window use vy-dc-h200-2 (in turn with H200-99, never beside its energy runs) or the new kernels H200 if it's named, or queue on -3 after 02:15Z.
