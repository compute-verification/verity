# Commitment hash

The kernels lane's workspace (`@kernels`): candidate commitment hashes for served inference, each measured on our GPUs and
judged against the binding notion it is proved to meet in Lean.

The goal is a hash, or a binding commitment under named assumptions, that brings vLLM's commit time down from today's
SHA-512 frame-v3 trees (the committer in `integrations/vllm/verity_vllm/commit/`). A candidate becomes the commitments'
hash only through its owners and with Daniel's ruling, because the commitment's hash is a design rule ("One hash",
AGENTS.md).

| Directory | What it measures |
|---|---|
| `model/` | commit overhead as a share of serve time, from committed bytes, leaf size and a hash's rate (`commit_model.py --card`) |
| `intpipe/` | SHA-512, SHA-256, BLAKE3 and reduced-round SHA-512 on the integer pipes, plain and in hm96 format, at prefill and decode window sizes, with the decode-latency levers |
| `sharaw/` | hm96-sha512 at 4 KiB leaves near the raw SHA-512 rate |
| `structure/` | leaf size, tree arity, salts off the step, and SHA-512 fused into the GEMM |
| `lattice/` | SIS (Ajtai) hashes on the INT8 tensor cores: parameters and throughput |
| `bytes/` | fewer SHA-512 compressions with the hash kept: values out of the Commit and leaf-format changes, priced through `model/` |
| `overlap/` | hm96-sha512 hashing of a Qwen3-8B prefill step beside its GEMMs (cuBLASLt, CUTLASS, a persistent hash server) |
| `loop/` | the standard loop: one iteration's served_commit runs on three GPUs (`iterate.sh`), their collection and labels (`collect.py`), and the plot from the evidence store (`plot.py`) |
| `maxtput/` | the Commit at vLLM's maximum throughput: native compiled vLLM at saturation against the Commit's arms on the same load, each a loop row |
| `dedup/` | Pearl-C's A rows read from a precomputed root against today's -h3, and the timing of compute accounting's -h4 branch |
| `memsat/` | memory accounting's remote audit at saturation (receipts and codec against plain serving), through `benchmarks/remote_session` |
| `dither/` | PoUW's H1 + fx1 receipts served in compiled vLLM on the RTX PRO 6000 (sm_120): the one-stream linear (`h1o.cu`) whose noised MMAs serve y, with the peel z = A Bᵀ from the receipted words (H1V3), which no credit gate on `main` reads yet; the A tables (`h1c.cu`); each kernel gated against CA's H1V3 (`h1ogate.py`), and the served overhead against stock vLLM int8, paired rep by rep (`sbench.py`) |
| `exactlin/` | the int8-exact linear as two Triton kernels (`exactlin.py`: the exact quantizer, and the GEMM with the exact epilogue and per-card configs), gated bit-exact against `int8_exact` (`kgate.py`) and the int8 Definitions' replay evaluators (`replay.py`), timed against stock vLLM int8 (Triton, CUTLASS, and CUTLASS on the transposed product, `kgate.cutlass_swapped`) and the exact path at kernel level (`kbench.py`, `ksweep.py`; the z/y record epilogue of `verity_int8::scaled_mm_zy` against the served product, `zybench.py`) and served (`served.sh`, through `dither/sbench.py`'s exact-fused arm, after the kernel gate), with the served y checked bit for bit by vllm's `int8_exact_gpu` (`values.py`), the per-step kernel table (`steps.py`) and the share of a served step PoUW doesn't credit (`share.py`) |

The campaign's findings and the recommendation to Daniel are in the notes, `lanes/hash/`.
