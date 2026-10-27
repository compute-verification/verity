# KSW inbox (kernels lead)

## 2026-10-27 23:45Z from kernels: HASH-50 moves up to Wed 12:00Z; a second H200 is coming
compute-accounting, who directs CMP-231, wants the fused orbit's wall time by Wed 28 Oct 12:00Z, not Thu: alone, and with SHA-512 at 0.51 of the bar beside it, against kernels' best int8 GEMM (wgmma.linear, or #5613's TMA-store GEMM) on 8B's four linears. Do the design and the CPU exactness check quickly, then the kernel.
- **GPU:** compute-accounting approved one more H200 for kernels, and infra is bringing it up tonight. When its name lands here, use it for HASH-50 first; it's yours and KU2's (HASH-51). Until then, vy-dc-h200-3 with short leases.
- **Overflow rates:** compute-research now measures real 8B form-overflow rates for C2-O (checkpoint weights and served activations, every linear, CPU, Wed 03:00Z, on CMP-231). Your kernel still needs B's overflow positions as metadata, so compute them for that. Don't publish a separate overflow study; cite theirs when it's up.
