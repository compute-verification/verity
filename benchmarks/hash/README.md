# Commitment hash

The kernels lane's workspace (`@kernels`): candidate commitment hashes for served inference, each measured on our GPUs and
judged against the binding notion it is proved to meet in Lean.

The goal is a hash, or a binding commitment under named assumptions, that brings vLLM's commit time down from today's
SHA-512 frame-v3 trees (the committer in `integrations/vllm/verity_vllm/commit/`). A candidate becomes the commitments'
hash only through its owners and with Daniel's ruling, because the commitment's hash is a design rule ("One hash",
AGENTS.md).

| Directory | What it holds |
|---|---|
| `lib/` | libvc, the served path's SHA-512 commitments on the GPU (`vc.h`, `vclib.py`), held to the CPU references by `conformance.py` |
| `dither/` | PoUW's H1 + fx1 receipts served in compiled vLLM on the RTX PRO 6000 (sm_120): the one-stream linear (`h1o.cu`) whose noised MMAs serve y, with the peel z = A Bᵀ from the receipted words (H1V3), which no credit gate on `main` reads yet; the A tables (`h1c.cu`); each kernel gated against CA's H1V3 (`h1ogate.py`), and the served overhead against stock vLLM int8, paired rep by rep (`sbench.py`) |
| `exactlin/` | the int8-exact linear as two Triton kernels (`exactlin.py`: the exact quantizer, and the GEMM with the exact epilogue and per-card configs), gated bit-exact against `int8_exact` (`kgate.py`) and the int8 Definitions' replay evaluators (`replay.py`), timed against stock vLLM int8 (Triton, CUTLASS, and CUTLASS on the transposed product, `kgate.cutlass_swapped`) and the exact path at kernel level (`kbench.py`, `ksweep.py`; the z/y record epilogue of `verity_int8::scaled_mm_zy` against the served product, `zybench.py`) and served (`served.sh`, through `dither/sbench.py`'s exact-fused arm, after the kernel gate), with the served y checked bit for bit by vllm's `int8_exact_gpu` (`values.py`), the per-step kernel table (`steps.py`) and the share of a served step PoUW doesn't credit (`share.py`) |
| `maxtput/build_job.sh` | a Build of record for a model that has none: circuits' served-match alone, as a `research run` job |

The campaign's findings and the recommendation to Daniel are in the notes, `lanes/hash/`, one note per measurement
(`hash/20261018T…-draft-<dir>`), each citing its runs. Its one-off harnesses (the cost model, the hash and pipe sweeps,
the tree-structure and lattice studies, the overlap, loop, max-throughput, dedup and memory-saturation runs) were deleted
once those notes held their results; Git's history keeps them (`git log --diff-filter=D -- benchmarks/hash`).
