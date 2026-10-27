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
