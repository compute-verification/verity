# Commitment hash

The hash lane's workspace (`@hash`): candidate commitment hashes for served inference, each measured on our GPUs and
judged against the binding notion it is proved to meet in Lean.

The goal is a hash, or a binding commitment under named assumptions, that brings vLLM's commit time down from today's
SHA-512 frame-v3 trees (the committer in `integrations/vllm/verity_vllm/commit/`, the GPU kernels in
`verity/ml/kernels/hash_gpu/`). A candidate moves into the soundness path only through its owners and with Daniel's
ruling, because the commitment's hash is a design rule ("One hash", AGENTS.md).
