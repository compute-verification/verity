# B-kernel P0 register dumps of the pinned FlashAttention-2 kernel (copied, unmodified)

Source: lane B-kernel (R8, 2026-09-18), worktree `/private/tmp/w-perf`, branch `lane/r8-fa2-kernel`, commit `0b5dbae4e`
("P0 GPU-tested + validated (ALL-OK) …"), files `out/gen/r8/fa2-kernel/results/p0/p0_*.npz` and `p0_summary.json`.
Copied here read-only on 2026-09-18 ~08:55Z by lane B-commit so that `test_kernel_dump.py` and
`out/gen/r8/fa2-commit/xcheck_kernel_dumps.py` stay reproducible after the R8 worktrees are gone.  `SHA256SUMS` lists
the digests of the copies (identical to the originals at copy time).

How they were produced (B-kernel STATUS/REPORT, source-inspected there): pod cp-fa2 (RTX 4090, sm_89 running the sm_80
cubin), vllm-flash-attn `506341a143fcabd4bb79052a7605ada727d6b3f5` rebuilt as a standalone torch extension with a
diagnostic tap (`verity_tap.patch`, `VERITY_TAP=2`) that stores, per kernel head and tile row, the live register values
`acc_s` raw / masked, `P` fp32, `rP` bf16, running `row_max`, rescale factor, step index, per-lane `row_sum` partials,
`acc_o` after every visited key block, and the final `row_sum` / `inv_sum` / `lse` / `row_max`, plus the bf16 inputs
(as fp32 copies), the paged block table, `seqused_k`, and the kernel's HBM `out` / `softmax_lse`.  The P0 build's `out`
and `lse` are bit-identical to the installed `_vllm_fa2_C.abi3.so` on every dumped shape.  Inputs are B-kernel's
synthetic bf16 tensors at the accepted run's shapes (and beyond), not the accepted record's own q/k/v.

| file | shape | kernel |
|---|---|---|
| `p0_16x32_h1kv1.npz` | 16 query × 32 keys, 1 head | causal, 1 block |
| `p0_64x128_h1kv1.npz` | 64 × 128 | causal, 1 block |
| `p0_40x200_h1kv1.npz` | 40 × 200 (chunk rows 160..199) | causal, 2 blocks |
| `p0_64x300_h1kv1.npz` | 64 × 300 (chunk rows 236..299) | causal, 3 blocks; rows with T ≤ 256 see block 2 fully masked |
| `p0_1x160_h1kv1.npz` | decode, 160 keys, 1 head | 2 blocks |
| `p0_1x160_h3kv1.npz` | decode, 160 keys, 3 query heads on 1 KV head (`seqlenq_ngroups_swapped`) | 2 blocks |
| `p0_1x169_h3kv1.npz` | decode, 169 keys, 3 query heads on 1 KV head | 2 blocks |

Reader: `verity_capture/experimental/fa2_commit/kernel_dump.py`.
