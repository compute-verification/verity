# PoUW benchmark harness

The shared harness for `protocols/pouw`'s schemes (`ncp-v1`, `pearl-fp8-v4`): one workload file, one set of kernels, the
same gates for every scheme. The cost model is W1 on an RTX 4090 (the PoUW problem statement §3.1).

| Script | Runs on | What |
|---|---|---|
| `gamma.py` | CPU | γ accounting: each scheme's honest reference priced term by term under W1, split into checked and unchecked work; Ω, γ where the conjecture names γ_0, γ's floor from unchecked work, and hashing per useful MAC |
| `gemm_bench.py` | RTX 4090 | Each scheme's honest GEMM against the plain GEMM of its datatype (cuBLASLt int8, FP8, BF16), at the workload's shapes. It first gates each kernel bit for bit against the `verity_pouw` reference, and refuses to time a kernel that fails |
| `route_u_bench.py` | RTX 4090 | Native NCP route-U gate against `checked()` on every 16x16 tile, then throughput against the signed-s8 checkpoint-dump kernel |
| `vllm_bench.py` | RTX 4090 | Overhead against plain vLLM (Qwen2.5-3B-Instruct): BF16, stock FP8, W7A7 (int7 without PoUW), NCP and Pearl, all by wrapping every `LinearBase` layer's `quant_method.apply`, the vLLM adapter's hook site. NCP's greedy tokens must equal W7A7's |
| `e2e_audit.py` | RTX 4090 | The protocol end to end: the GPU kernels prove, the pure-Python reference verifies the work-weighted draws; an honest prover is accepted, and a prover that corrupts half the tiles is rejected |
| `pouw_bench.py` | CPU (`--dry-run`), RTX 4090 | PoUW alone on fixed matmul shapes (`pouw_bench_shapes.json`: #389's and #435's Qwen2.5-0.5B linears, census workloads' GEMMs by id, and 4,096³ and 8,192³), with no served model. `ncp-v2`'s fused prover per call against the fastest plain path of the same shape (int8 serving, FP8 at half the int8 rate on the 4090, BF16; each at the better of cuBLASLt's default and Inductor's autotuned call), with hashing and XOF reported separately (committed less the no-hash control), and change 1's async path. Every arm is timed the same way: bursts on CUDA events over operand sets past twice the L2, a read flush with no dirty lines, the arms rotated each rep, every rep saved with clocks, power and versions. Every timed shape and flag set is gated on its root against the independent device build before it is timed, with a weight id per operand set, and on a consistency floor after. W1's W_ref/mkn beside Ω and γ; samples per bit with the verifier's cost per draw (our definition). Each variant of the tile kernel (`--tile-variants`) is gated against `pouw_native` on the CPU twin and on the device first |
| `kernels/pouw_gemm.cu` | RTX 4090 | NCP's native u8xs8 route-U GEMM from `c0`, the signed-s8 control, both with every 16-deep checkpoint written out, and Pearl's e4m3 GEMM on the `ada` device |
| `workloads.json` | | The model and its linear shapes, the decode batches and prefill lengths, and the reference shapes |

~~~bash
uv run python benchmarks/pouw/gamma.py                                   # CPU
# on a 4090 pod, with torch and vLLM, verity and verity_pouw on PYTHONPATH:
python benchmarks/pouw/gemm_bench.py --out gemm.json
python benchmarks/pouw/route_u_bench.py --out route_u.json
VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/pouw/vllm_bench.py --model /path/to/Qwen2.5-3B-Instruct --out vllm.json
python benchmarks/pouw/e2e_audit.py --out e2e.json
python benchmarks/pouw/pouw_bench.py --out pouw_bench.json     # CPU-only parts first: --dry-run
~~~

What the timings cover:

- **Noise and hashing.** The timed vLLM modes draw noise from torch's generator. SHAKE256 over NCP's transcript
  (3k/16 words per output) is costed at the research store's measured 650 GB/s, not run in the timed path.
- **Pearl's operand steps.** Pearl's noisy quantization and peel run as unfused torch operations. Pearl's own fused
  SM100 kernel is not portable to the 4090.
- **The model.** Qwen2.5-3B has every linear inside both schemes' domains. Qwen2.5-0.5B's k = 896 is below them, and
  Qwen2.5-1.5B's n = 1536 puts NCP's γ above 1%.

Native route-U result (`r20260928-120218-f7d7`, RTX 4090, m=16, k=2048, n=256):
all 16 tile digests and outputs match `ncp-v1.checked()`, the final int32 output is exact, and checkpoint-dump time is
1.359 ms against 1.348 ms for signed s8 (1.0084× time, 0.9917× throughput).  SASS run `r20260928-120439-c053` contains
native `IMMA.16816.U8.S8` and the signed `IMMA.16816.S8.S8` control.
