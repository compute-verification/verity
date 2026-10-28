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
