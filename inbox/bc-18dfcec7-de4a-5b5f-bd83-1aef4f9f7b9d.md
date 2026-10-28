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
