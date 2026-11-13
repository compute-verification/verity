[2026-11-13T01:51Z] d870654e3dc7
TOPCODE, Thu 5:52 PM PST

**With `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` set, bf16 causal SDPA at 4 threads still returns a different value for one tile of the same input: 4 calls in 2,000. At 1 thread none differ in 2,000, with the limit or without it.**

Machine (RAN): my VM, not a node. `lscpu | rg 'Model name'`: `Intel(R) Xeon(R) Processor`; `nproc`: 4; flags include amx_bf16, amx_tile and avx512_bf16. torch 2.14.0+cpu, from `uv run --project ../.. --extra torch-cpu` in `benchmarks/training_slowdown`.

The script (RAN, just now; I can commit it wherever you want it):

~~~python
import os, sys, torch, torch.nn.functional as F
threads, calls = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(threads)
torch.manual_seed(0)
q, k, v = (torch.randn(2, 2, 1024, 8).bfloat16() for _ in range(3))
base = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# then `calls` more calls on the same q, k, v; each output not torch.equal to base is counted, with where it differs
~~~

The inputs are q, k and v of shape [2, 2, 1024, 8] (batch, heads, tokens, head dim), bf16, from `torch.randn` under seed 0, causal and with no mask. That is the synthetic model's attention shape at one row a forward × 2 rows.

| `ONEDNN_MAX_CPU_ISA` | threads | calls | differing | where |
|---|---|---|---|---|
| unset | 4 | 2,000 | 15 | each in one (batch, head), one run of 16 or 32 tokens, e.g. tokens 1008..1023 of (1, 1), max \|diff\| 0.041; once 382 entries over tokens 48..223 of (1, 0) and (1, 1), max 0.53 |
| AVX512_CORE_BF16 | 4 | 2,000 | 4 | (0, 1) tokens 448..479, max 0.057; (1, 0) 512..543, max 0.090; (1, 0) 288..319, max 0.085; (0, 1) tokens 0..31, max 1.77 |
| unset | 1 | 2,000 | 0 | |
| AVX512_CORE_BF16 | 1 | 2,000 | 0 | |

Where I first saw it: `det8.py`, the same loop over 400 calls, in a plain `python` on my VM with `ONEDNN_MAX_CPU_ISA` unset (RAN; one call in 400 differed). Not set there: my shell's environment has no `ONEDNN_*` or `OMP_*` variable (RAN, `compgen -e`). The slow resume test's divergence at step 2 was in the processes `run.main` starts, which go through `run.limit_cpu`, so with the limit on (INFERRED from your note on `limit_cpu`; I didn't print the variable inside them). The table's second row shows the limit doesn't stop it (RAN).

INFERRED: each differing output is wrong in one 16- or 32-token block of one (batch, head), the size of an SDPA kernel's tile. A max |diff| of 1.77 is far beyond reordering a bf16 sum, so this looks more like a race between threads in the CPU flash-attention kernel than like summation order.

In flight: every-entry of_record at GROW's head b16502a23d (r20261113-015001-f3a4) and at mine, dea3ccc1a1 (r20261113-015028-3b8e); the 238M trace (r20261113-014549-a81c).

