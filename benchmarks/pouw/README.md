# PoUW benchmark harness

The shared harness for `protocols/pouw`'s schemes (`ncp-v1`, `pearl-fp8-v4`): one workload file, one set of kernels, the
same gates for every scheme. The cost model is W1 on an RTX 4090 (the PoUW problem statement §3.1).

| Script | Runs on | What |
|---|---|---|
| `gamma.py` | CPU | γ accounting: each scheme's honest reference priced term by term under W1, split into checked and unchecked work; Ω, γ where the conjecture names γ_0, γ's floor from unchecked work, the bytes hashed per useful MAC, and the overhead against the plain GEMM per binding (none, and each hash in its `HASHES` table), with γ "—" where nothing is bound |
| `gemm_bench.py` | RTX 4090 | Each scheme's honest GEMM against the plain GEMM of its datatype (cuBLASLt int8, FP8, BF16), at the workload's shapes. It first gates each kernel bit for bit against the `verity_pouw` reference, and refuses to time a kernel that fails |
| `route_u_bench.py` | RTX 4090 | Native NCP route-U gate against `checked()` on every 16x16 tile, then throughput against the signed-s8 checkpoint-dump kernel |
| `vllm_bench.py` | RTX 4090 | Overhead against plain vLLM (Qwen2.5-3B-Instruct): BF16, stock FP8, W7A7 (int7 without PoUW), NCP and Pearl (as shipped and as its whitepaper describes it), all by wrapping every `LinearBase` layer's `quant_method.apply`, the vLLM adapter's hook site. NCP's greedy tokens must equal W7A7's |
| `e2e_audit.py` | RTX 4090 | The protocol end to end: the GPU kernels prove, the pure-Python reference verifies the work-weighted draws; an honest prover is accepted, and a prover that corrupts half the tiles is rejected |
| `kernels/pouw_gemm.cu` | RTX 4090 | NCP's native u8xs8 route-U GEMM from `c0`, the signed-s8 control, both with every 16-deep checkpoint written out, and Pearl's e4m3 GEMM on the `ada` device |
| `workloads.json` | | The model and its linear shapes, the decode batches and prefill lengths, and the reference shapes |

~~~bash
uv run python benchmarks/pouw/gamma.py                                   # CPU
# on a 4090 pod, with torch and vLLM, verity and verity_pouw on PYTHONPATH:
python benchmarks/pouw/gemm_bench.py --out gemm.json
python benchmarks/pouw/route_u_bench.py --out route_u.json
VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/pouw/vllm_bench.py --model /path/to/Qwen2.5-3B-Instruct --out vllm.json
python benchmarks/pouw/e2e_audit.py --out e2e.json
~~~

What the timings cover:

- **Hashing is in the headline.** Hash calls are free in γ (an adversary with free hashing is only stronger), but the
  honest prover pays for them, so every headline slowdown (`*_total_*`, `timing_total`) is the arithmetic plus the
  binding: every checked word (NCP: 3k/16 per output, 0.75 B per useful MAC) hashed at the scheme's pinned hash
  (`gamma.PINNED`: SHAKE256 for `ncp-v1`, BLAKE3 for `pearl-fp8-v4`). The digest is not run in the timed paths yet; it is
  added as serial time at the device rates in `gamma.HASHES`. SHAKE256 is measured on an RTX 4090 (650 GB/s,
  525 units per byte); SHA-256, TurboSHAKE128 and BLAKE3 are estimated from it. The hash-free figures (`*_arith_*`,
  `*_kernel_*`, `timing`) are components: they have no binding, so they carry no γ. The noise XOF and the activation
  commitment add about 2/n bytes per MAC and are not counted.
- **Noise.** The timed vLLM modes draw noise from torch's generator.
- **Pearl's operand steps.** Pearl's noisy quantization and peel run as unfused torch operations. Pearl's own fused
  SM100 kernel is not portable to the 4090.
- **Pearl's two readings (`gamma.PEARL_READINGS`).** Pearl's sources disagree on F_A, so every table reports both, and
  each is labelled. The headline is **as shipped** (`pearl-fp8-v4 (as shipped)`, `pearl_*`, vLLM mode `pearl`): F_A is
  keyed by seed_B, as Pearl's consensus code and `schemes/pearl_kw` run it, so B̃·F_A is paid once per weight. The
  **whitepaper** reading (`pearl-fp8-v4 (whitepaper)`, `pearl_whitepaper_*`, vLLM mode `pearl-wp`) draws F_A fresh for
  every activation commitment (§4.5, eq. 10), so B̃·F_A is paid per unit. At 8192³ the γ floor is 2.312% as shipped
  and 3.044% in the whitepaper reading, which grows as r/m (79.9% at m = 16).
- **The model.** Qwen2.5-3B has every linear inside both schemes' domains. Qwen2.5-0.5B's k = 896 is below them, and
  Qwen2.5-1.5B's n = 1536 puts NCP's γ above 1%.

Native route-U result (`r20260928-120218-f7d7`, RTX 4090, m=16, k=2048, n=256):
all 16 tile digests and outputs match `ncp-v1.checked()`, the final int32 output is exact, and checkpoint-dump time is
1.359 ms against 1.348 ms for signed s8 (1.0084× time, 0.9917× throughput).  SASS run `r20260928-120439-c053` contains
native `IMMA.16816.U8.S8` and the signed `IMMA.16816.S8.S8` control.
