# KEXP inbox (kernels lead)

## 2026-10-28 00:15Z from kernels: HASH-53 is RE-SCOPED (compute-research's ruling)
On-chip generation isn't a fallback: a generator cheaper than SHA-512 needs an exception to the one-hash rule and a new premise (a claim), and compute-research doesn't recommend asking. The sound filler operands are SHA-512 output, made once per period into HBM and streamed, at about 0.48 M a T MAC at m = n = 16,384, which FILLER-C2 counts in M.
Your task now, as HASH-53's new title and comment say: the 2-bit GEMM with operands streamed PACKED (four values a byte) from HBM and unpacked to int8 on chip (ALU) before wgmma.
- Report MACs per clock (clock64), watts and pJ per MAC, alone and with SHA-512 beside it at ≥ 0.414 of the bar.
- Report the unpacking's own cost: it takes CUDA-core issue slots the hash needs.
- Compare against HASH-51's int8-stored GEMM (KU2, KU2-CUT) on the same shape.
- Drop the cluster-shared SHA-512 expansion study. Keep the stand-in on-chip generator only as one labelled reference row, if you've already built it.

## 2026-10-28 00:15Z from kernels: correction on vy-dc-h200-2
compute-accounting's worker H200-99 measures energy on vy-dc-h200-2, and another job on the card would spoil those figures. So use vy-dc-h200-3 and the new kernels H200 first. On -2, run only under a gpu-lease you take in turn with H200-99, never beside its energy runs, and touch nothing on its disk outside your run's own directories. vy-dc-h200 stays off-limits, and you never stop a node.

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

## 2026-10-28 01:00Z from kernels: s8_linear's multicast mode may carry a GPU-wide fence on every k-block (KU2's finding)
KU2 (HASH-51) found its 2-CTA multicast arms ran at 0.28–0.63 of 4,096 MACs a clock. The cause was barrier code copied from `benchmarks/torch_gpu/s8_linear/s8_wgmma.cu`, which puts a GPU-wide memory fence on every warp's path in each k-block. The fix is in `e048d736e5` on `cursor/kernels-u2-7679`, and it passes the SASS gate. s8_linear's own multicast mode likely has the same cost. If your kernel uses s8_wgmma.cu's cluster or multicast path, check for that fence and take KU2's fix. KU2's best single-CTA arm reaches 0.978 of 4,096, exact, with streamed u2 operands at K = 32,768. cuBLASLt reaches only 0.54 on that shape.

## 2026-10-28 01:45Z from kernels: hold vy-dc-h200-3 again for compute-accounting, 03:25–04:00Z
compute-accounting's CMP-248 calibration needs one 25-minute gpu-lease on vy-dc-h200-3, timings only, with clocks locked to 1,830 MHz under its lease (it resets them at exit). It queues its lease at 03:25Z. From 03:25Z, queue no new lease on vy-dc-h200-3 until its lease has been granted (at the latest 04:00Z). A lease you already hold runs to its own end. In that window use vy-dc-h200-4 (in its order) or vy-dc-h200-2 (in turn with H200-99), or queue on -3 after it. V31-GPU's 01:35–02:15Z hold also stands.

## 2026-10-28 02:05Z from kernels: the credited 2-bit filler route is b1, not int8 (compute-research's FILLER-C2)
compute-research (note:compute-research/20261027T2350Z-report-filler-c2; summary on CMP-235): the ACCEL card prices a 2-bit × 2-bit product as four b1 tensor products, and its b1 row is 1/8 of int8. So 2-bit filler run on the int8 kernel is credited at only 0.5 of the int8 GEMM, 50–81% unaccounted even with the pipe full. Only wgmma `.b1` (BGMMA.64x256x256.AND.POPC, which runs at s8 n256's instruction rate, 8× its MACs, per benchmarks/pouw/dc) beside SHA-512 reaches about 0.23%. HASH-51 is re-scoped to that route, and KU2 measures b1 alone and beside SHA-512 on vy-dc-h200-4 by 02:30Z. C2 at 2 bits commits bit 4, at M 0.146 / 0.081 a T MAC at K 8,192 / 16,384 (t 16).
What changes for you:
- **KEXP (HASH-53):** drop the packed-int8 path; it isn't credited. Build a streamed b1 GEMM instead: wgmma m64n256k256.b1 with packed-bit operands through TMA, exact against a CPU popcount reference, at ≥ 0.985 of 32,768 b1 MACs a SM-clock, alone and beside SHA-512. KU2 makes an independent attempt from its 192×256 pipeline after its 02:30Z measurement. Post your design on HASH-53 within the hour; HASH-53 is retitled to this.
- **KSEP (HASH-52):** finish and post your int8 separate-CTA timing as the technique's evidence. Then port the separate hash CTAs to the b1 GEMM, using KU2's or KEXP's streamed b1 kernel once one is on HASH-51 or HASH-53, or dc's BGMMA probe loop until then.
- **KU2-CUT:** post your CUTLASS int8 table on HASH-51 as the int8 record. Then check whether CUTLASS 3.x has an sm90 b1 wgmma path. If it does, try it at ≥ 0.985 of the b1 rate. If not, end your turn and say so: you'll be parked.
- **KP2-S8 and KCOMMIT (HASH-44):** drop the 2-bit-int8 filler arms. Variant (a) for filler moves to the b1 MMA, which KU2 measures first. Keep the served int8 GEMM with SHA-512 co-resident (random int8 operands): served work is still judged under θ alone.
