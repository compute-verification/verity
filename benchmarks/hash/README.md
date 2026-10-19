# Commitment hash

The hash lane's workspace (`@hash`): candidate commitment hashes for served inference, each measured on our GPUs and
judged against the binding notion it is proved to meet in Lean.

The goal is a hash, or a binding commitment under named assumptions, that brings vLLM's commit time down from today's
SHA-512 frame-v3 trees (the committer in `integrations/vllm/verity_vllm/commit/`, the GPU kernels in
`verity/ml/kernels/hash_gpu/`). A candidate moves into the soundness path only through its owners and with Daniel's
ruling, because the commitment's hash is a design rule ("One hash", AGENTS.md).

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

The campaign's findings and the recommendation to Daniel are in the notes, `lanes/hash/`.
