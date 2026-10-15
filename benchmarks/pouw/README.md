# PoUW benchmark harness

The shared harness for `verity/core/protocols/pouw`'s schemes (`ncp-v1`, `pearl-fp8-v4`): one workload file, one set of kernels, the
same gates for every scheme. The cost model is W1 on an RTX 4090 (the PoUW problem statement §3.1).

| Script | Runs on | What |
|---|---|---|
| `gamma.py` | CPU | γ accounting: each scheme's honest reference priced term by term under W1, split into checked and unchecked work; Ω, γ where the conjecture names γ_0, γ's floor from unchecked work, and hashing per useful MAC |
| `gemm_bench.py` | RTX 4090 | Each scheme's honest GEMM against the plain GEMM of its datatype (cuBLASLt int8, FP8, BF16), at the workload's shapes. It first gates each kernel bit for bit against the `verity.core.protocols.pouw` reference, and refuses to time a kernel that fails |
| `route_u_bench.py` | RTX 4090 | Native NCP route-U gate against `checked()` on every 16x16 tile, then throughput against the signed-s8 checkpoint-dump kernel |
| `vllm_bench.py` | RTX 4090 | Overhead against plain vLLM (Qwen2.5-3B-Instruct): BF16, stock FP8, W7A7 (int7 without PoUW), NCP and Pearl, all by wrapping every `LinearBase` layer's `quant_method.apply`, the vLLM adapter's hook site. NCP's greedy tokens must equal W7A7's |
| `e2e_audit.py` | RTX 4090 | The protocol end to end: the GPU kernels prove, the pure-Python reference verifies the work-weighted draws; an honest prover is accepted, and a prover that corrupts half the tiles is rejected |
| `kernels/pouw_gemm.cu` | RTX 4090 | NCP's native u8xs8 route-U GEMM from `c0`, the signed-s8 control, both with every 16-deep checkpoint written out, and Pearl's e4m3 GEMM on the `ada` device |
| `pearl_c/` | CPU | Pearl-C's hashing format -h3 on the device (`hash_h3.cuh`: SHA-512 row digests, the `audit.HM96` trees, the row and unit seeds with their line keys, and the tile leaves, which `pearl_c_sm120.cu` includes), its host build (`hash_host.cpp`, `simt_twin.py`, thread for thread, for byte tests without a GPU) and `fake_cuda.c`, the stand-in for libcuda that `run.py`'s dry run uses. The H100 -h1 bench that lived here, and every BLAKE3 kernel, are in `archive/pouw-blake3/` (one hash, SHA-512: Daniel, 2026-10-06) |
| `pearl_c_sm120/` | RTX PRO 6000 Blackwell | `pearl-c-sm120-v1` (G = 4) and the unpromoted record on sm_120a: the same pipeline on `mma.sync` (`kind::f8f6f4` m16n8k32 E4M3 atoms, m16n8k16 BF16 peel) fed by `ldmatrix` from `cp.async`-filled swizzled shared memory within 99 KB, row-major throughout. `build.sh` builds the cubin with nvcc ≥ 12.8 and refuses it unless `sass_gate.py` finds every atom as `QMMA.16832.F32.E4M3.E4M3`, every peel step as `HMMA.16816.F32.BF16` and no local memory; `fixture.py` writes the check on both device records. `run.py` refuses any device but the expected UUID (NVML by UUID), checks every buffer bit for bit, then times diagnostic phases; the panel's -h1 arm (`pearlc_arm.py`, `verify.py`) and the salt-to-deadline exhaustion window are in `archive/pouw-blake3/` until they are ported to -h3 |
| `pearlc_census.py` | CPU | TT_OUT's falsification census for Pearl-C's ticket chain, bit-exact: per tile, the atoms an adversary could skip (tile identities, exact runs, no-op promotion adds) over input families built to trigger each, on the H100's atom or sm_120's (`--atom`), and the debit for joint skips; `--selftest` checks the vectorized atom against `verity.ml.circuits.silicon.models` |
| `rs_attacks/` | CPU, RTX PRO 6000 Blackwell | Attacks on the committed prefix targets of (A + UVᵀ)(B + WZᵀ) with rank-r masks: `attacks.py` checks each route bit for bit and prices it (ledger and time), with the threshold r* (`check`, `table`); `prefix.cu` (`prefix.sh`, under `gpu-lease`) times the honest int8 step against the CUDA-core routes on one GPU |
| `workloads.json` | | The model and its linear shapes, the decode batches and prefill lengths, and the reference shapes |

~~~bash
uv run python benchmarks/pouw/gamma.py                                   # CPU
# on a 4090 pod, with torch and vLLM, verity and verity.core.protocols.pouw on PYTHONPATH:
python benchmarks/pouw/gemm_bench.py --out gemm.json
python benchmarks/pouw/route_u_bench.py --out route_u.json
VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/pouw/vllm_bench.py --model /path/to/Qwen2.5-3B-Instruct --out vllm.json
python benchmarks/pouw/e2e_audit.py --out e2e.json
~~~

What the timings cover:

- **Noise and hashing.** The timed vLLM modes draw noise from torch's generator. SHAKE256 over NCP's transcript
  (3k/16 words per output) is costed at a measured 650 GB/s, not run in the timed path.
- **Pearl's operand steps.** Pearl's noisy quantization and peel run as unfused torch operations. Pearl's own fused
  SM100 kernel is not portable to the 4090.
- **The model.** Qwen2.5-3B has every linear inside both schemes' domains. Qwen2.5-0.5B's k = 896 is below them, and
  Qwen2.5-1.5B's n = 1536 puts NCP's γ above 1%.
- **The served baseline.** Served claims (`pearl_c_vllm/`) are measured against graphed stock FP8, the baseline: stock
  vLLM `quantization="fp8"` compiled with its CUDA graphs, `e2e.py`'s ``fp8-graphs`` mode (``--fp8-graphs``).

Native route-U result (`r20260928-120218-f7d7`, RTX 4090, m=16, k=2048, n=256):
all 16 tile digests and outputs match `ncp-v1.checked()`, the final int32 output is exact, and checkpoint-dump time is
1.359 ms against 1.348 ms for signed s8 (1.0084× time, 0.9917× throughput).  SASS run `r20260928-120439-c053` contains
native `IMMA.16816.U8.S8` and the signed `IMMA.16816.S8.S8` control.
