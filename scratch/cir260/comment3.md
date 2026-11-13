**The FP8 cache-write capture ran a second time on an H200, at PR [#9316](https://github.com/compute-verification/verity/pull/9316)'s head `ffaacd91c9dd`, as run `r20261113-031922-52f4`: 12,512 cache rows of 656 bytes compared with `DsaLatentRowDsMla_v1`, 0 rows differ. The kernel's inputs and outputs are byte for byte the first try's.**

This is the same capture as `r20261112-172953-93dc`, which ran at `f912fe226f80`. The head since changed only how the script reports its own errors, so this run checks that the script still gives the same answer.

Rows compared, and rows that differ:
- cache rows: 3,128 per case and specialisation, 12,512 in all; 0 differ, in any of their 656 bytes;
- `kv_c_out` and `k_pe_out` against the Definition's `latent`: 3,136 tokens per case and specialisation, 12,544 in all; 0 differ;
- the norm words against `DsaRMSNorm_v1{N=512}` and the RoPE words against `RoPEHeadGptJ_v2{D=64}`, as registered: 0 differ on any token.

What else the run shows:
- **The verdict** is GO, with no error in either phase.
- **Repeats.** Each case ran twice per specialisation with the same bytes, the two specialisations gave the same rows, and every cache byte outside the written rows kept its `0xA5` poison.
- **The compiled PTX** of both FP8 specialisations has the code sha256 the Definition's docstring cites.
- **Corners reached** are the first try's counts exactly. In the corner case: 154 groups of zeros at the floor, 37 groups below FLT_MIN, 223 groups with an infinite scale, 2,905 NaN values in finite groups, 3,351 `0x7F` bytes, 199,294 `0x80` bytes and 12,098 bytes at ±448.
- **Against the first try.** All 20 saved word files (the kernel's inputs, and its outputs in both cases and both specialisations) have the same sha256 as the first try's.

Where and how it ran:
- node A, card 2 (NVIDIA H200, 132 SMs), cores 20-29, vLLM `0.28.1rc1.dev472+gd9105ea80`, inside circuits' row of 03:15Z to 04:25Z;
- the start check found card 2 idle at 03:20:01Z; the lease was granted at once, declared cores 20-29, and held the card 11 s;
- the run took 64 s (03:19:57Z to 03:21:01Z), from a clean tree at `ffaacd91c9dd`. Card 2 is free.

The PR merged at 02:13:43Z, before this run. The Definition modules, their sidecars, the hand-vector test, the capture script and its test are the same bytes on `main` (`1e4d5c993c`) as at `ffaacd91c9dd`, so this run speaks for what landed.

Not established here, as before: served vLLM end to end with `fp8_ds_mla`, FlashMLA's read (CIR-262), the dense prefill read-back (CIR-261), and a sealed record.
