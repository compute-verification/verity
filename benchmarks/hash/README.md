# Commitment hash

The hash lane's workspace (`@hash`): candidate commitment hashes for served inference, each measured on our GPUs and
judged against the binding notion it is proved to meet in Lean.

The goal is a hash, or a binding commitment under named assumptions, that brings vLLM's commit time down from today's
SHA-512 frame-v3 trees (the committer in `integrations/vllm/verity_vllm/commit/`, the GPU kernels in
`verity/ml/kernels/hash_gpu/`). A candidate becomes the commitments' hash only through its owners and with Daniel's
ruling, because the commitment's hash is a design rule ("One hash", AGENTS.md).

| Directory | What it measures |
|---|---|
| `model/` | commit overhead as a share of serve time, from committed bytes, leaf size and a hash's rate (`commit_model.py --card`) |
| `intpipe/` | SHA-512, SHA-256, BLAKE3 and reduced-round SHA-512 on the integer pipes, plain and in hm96 format, at prefill and decode window sizes, with the decode-latency levers |
| `dither/` | PoUW's H1 + fx1 receipts served in compiled vLLM on the RTX PRO 6000 (sm_120): the one-stream linear (`h1o.cu`) whose noised MMAs serve y, with the credited peel z = A Bᵀ; the A tables (`h1c.cu`); each kernel gated against CA's H1V3 (`h1ogate.py`), and the served overhead against stock vLLM int8, paired rep by rep (`sbench.py`) |
| `sharaw/` | hm96-sha512 at 4 KiB leaves near the raw SHA-512 rate |
| `structure/` | leaf size, tree arity, salts off the step, and SHA-512 fused into the GEMM |
| `lattice/` | SIS (Ajtai) hashes on the INT8 tensor cores: parameters and throughput |

The campaign's findings and the recommendation to Daniel are in the notes, `lanes/hash/`.
