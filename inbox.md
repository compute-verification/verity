[2026-11-12T23:47Z] d201f02fb59a
BACKWARD2, Thu 12 Nov 3:48 PM PST.

**The probe ended with rc 0, and its overflow case contradicts must 1 as the Definition states it: on the card every dQ and dK word of that case is non-finite, where the fixed Definition (the dP sum FMUL then FFMA, dim 0's product first) gives every word finite.**

- RAN, r20261112-234502-9972, rc 0, on vy-nebius-2: it took its slot at once, and the lease held GPU 0 for 19 s. Last line: "gpu-lease: GPU 0 (GPU-7833ad2c) held 0m19s, busy 0m00s of 0m19s sampled (0%)".
- Its two result lines:
  - "bwd/probe/det/ftz {"repeat_differ_words": [0, 0, 0], "nan_or_inf_words": {"dq": 0, "dk": 0, "dv": 0}}"
  - "bwd/probe/det/ovf {"repeat_differ_words": [0, 0, 0], "nan_or_inf_words": {"dq": 3145728, "dk": 3145728, "dv": 0}}"
- Kernels: `flash_bwd_dot_do_o_kernel`, `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` (grid [4, 2, 24]) and `flash_bwd_convert_dq_kernel`, the capture's own.
- What the overflow case is: on every row, dO's dims 0 and 1 are 2^63 and 2^64 and O's are -2^64 and 2^64, so thread 0's products are -2^127 and 2^128. FMUL then FFMA in dim order sums them to a finite 2^127. Anything that rounds the 2^128 product alone gives inf, and with it every dQ and dK word is non-finite. That covers FADD after FMUL, as the pre-review head had it, and also any order that takes dim 1's product before dim 0's. The card gives every word non-finite. So the card's dot_do_o is not "dim 0 by FMUL, then FFMA in dim order", or it doesn't read thread 0's dims in that order.
- What it does not settle (INFERRED): FADD and a different first product both overflow here, so this case can't tell them apart. Every captured word fits both the old head (FADD) and the fixed head (FFMA). So the fixed head's Pre rests on the SASS reading, and this probe says that reading, as the Definition writes it, misses these words.
- Next, unless you say otherwise: the planned CPU fit of the probe's 8 heads through the Definitions, at the fixed head `a809ac0598` and at the pre-review head `84e7fc9161` (FADD), on the CPU pool. Each word of dQ, dK and dV (NaN bits included) is compared with the card's. That shows whether the FADD Pre gives the card's words where the FFMA Pre doesn't, and what the `ftz` case's words say about the atomics' FTZ. I push nothing to the Definitions until you rule.

