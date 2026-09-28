# PoUW benchmark harness

The shared harness for `protocols/pouw`'s schemes (`ncp-v1`, `pearl-fp8-v4`): one workload file, one set of kernels, the
same gates for every scheme. The cost model is W1 on an RTX 4090 (the PoUW problem statement §3.1).

| Script | Runs on | What |
|---|---|---|
| `gamma.py` | CPU | γ accounting: each scheme's honest reference priced term by term under W1, split into checked and unchecked work; Ω, γ where the conjecture names γ_0, γ's floor from unchecked work, and hashing per useful MAC |
| `gemm_bench.py` | RTX 4090 | Each scheme's honest GEMM against the plain GEMM of its datatype (cuBLASLt int8, FP8, BF16), at the workload's shapes. It first gates each kernel bit for bit against the `verity_pouw` reference, and refuses to time a kernel that fails |
| `vllm_bench.py` | RTX 4090 | Overhead against plain vLLM (Qwen2.5-3B-Instruct): BF16, stock FP8, W7A7 (int7 without PoUW), NCP and Pearl, all by wrapping every `LinearBase` layer's `quant_method.apply`, the vLLM adapter's hook site. NCP's greedy tokens must equal W7A7's |
| `e2e_audit.py` | RTX 4090 | The protocol end to end: the GPU kernels prove, the pure-Python reference verifies the work-weighted draws; an honest prover is accepted, and a prover that corrupts half the tiles is rejected |
| `kernels/pouw_gemm.cu` | RTX 4090 | NCP's s8 GEMM with every 16-deep checkpoint read (fold) or written out (dump), and Pearl's e4m3 GEMM on the `ada` device (a fresh accumulator every 128 products, promoted by one FP32 add), JIT-built as a torch extension |
| `workloads.json` | | The model and its linear shapes, the decode batches and prefill lengths, and the reference shapes |

~~~bash
uv run python benchmarks/pouw/gamma.py                                   # CPU
# on a 4090 pod, with torch and vLLM, verity and verity_pouw on PYTHONPATH:
python benchmarks/pouw/gemm_bench.py --out gemm.json
VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/pouw/vllm_bench.py --model /path/to/Qwen2.5-3B-Instruct --out vllm.json
python benchmarks/pouw/e2e_audit.py --out e2e.json
~~~

What the timings cover:

- **Noise and hashing.** The timed vLLM modes draw noise from torch's generator. SHAKE256 over NCP's transcript
  (3k/16 words per output) is costed at the research store's measured 650 GB/s, not run in the timed path.
- **Pearl's operand steps.** Pearl's noisy quantization and peel run as unfused torch operations. Pearl's own fused
  SM100 kernel is not portable to the 4090.
- **The model.** Qwen2.5-3B has every linear inside both schemes' domains. Qwen2.5-0.5B's k = 896 is below them, and
  Qwen2.5-1.5B's n = 1536 puts NCP's γ above 1%.
