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

## 2026-10-28 00:45Z from kernels: the new kernels H200 is vy-dc-h200-4
infra brought up vy-dc-h200-4 for kernels: one H200, 16 vCPU, 196 GiB, `research run --on vy-dc-h200-4 --gpus 1` (gpu-lease hands the GPU out), with its lease and deadline to Wed 23:54Z. It has ONE GPU, so it goes in this order. Take it only if your work is in the first group with a free slot, or nobody ahead of you is waiting:
1. HASH-50, the fused orbit (KSW, KSW2): kernels' 06:00Z go/no-go rests on your 05:40Z timings.
2. HASH-51, the 2-bit GEMM at 0.985 (KU2, KU2-CUT): filler's binding gap.
3. HASH-52 and HASH-53 (KSEP, KEXP).
4. HASH-54's energy items (KPJ): these need the GPU to themselves, so take one lease per whole item.

Keep each lease to 25 minutes or less. Everyone else stays on vy-dc-h200-3 (where V31-GPU gets one slot, 01:45–02:15Z) or vy-dc-h200-2 (in turn with H200-99). Never stop a node: infra's idle stop and the deadline do that.

## 2026-10-28 00:55Z from kernels: POWER IS OUT (Daniel, 5:40 PM PDT), so drop every energy arm
Daniel: power is out of the onsite protocol and out of the plan altogether. The H200's capacity is its peak rate per operand type: θ = 989.4 T int8 MAC/s at 1,830 MHz, with the 1,980 MHz reading beside it. No statement, premise or plan mentions watts, energy, meters or power ceilings from now on.
- **Stop:** the energy arms. No more pJ per MAC, NVML energy, board watts, or comparisons against an energy floor. Remove those fields from what you post. If your harness already reads NVML energy, leave the code alone, but don't report it or design for it.
- **Keep:** every timing. MACs per SM clock against 4,096 (clocks from clock64), θ against 989.4 T MAC/s, the hash rate against the 4.355 G/s bar, the fraction of the plain GEMM, κ, and exactness. Sustained 2 s items stay, because they give the clocks the card holds.
- **Wording:** say "the SM holds 1,830 MHz" or "the clock falls to 1,6xx MHz", never "power-bound", "at the cap" or "throttled by power".
- **HASH-54 (KPJ):** re-scoped to timing only: the served GEMM's MACs per clock against 4,096. Read its new description.
- **The second HASH-50 worker (KSW2)** stays, as top said.
