# Commitment hash

The hash lane's workspace (`@hash`): candidate commitment hashes for served inference, each measured on our GPUs and
judged against the binding notion it is proved to meet in Lean.

The goal is a hash, or a binding commitment under named assumptions, that brings vLLM's commit time down from today's
SHA-512 frame-v3 trees (the committer in `integrations/vllm/verity_vllm/commit/`, the GPU kernels in
`verity/ml/kernels/hash_gpu/`). A candidate moves into the soundness path only through its owners and with Daniel's
ruling, because the commitment's hash is a design rule ("One hash", AGENTS.md).

- `memsat/`: the Commit + memory arm at saturation for the int8 Llama-3.1-8B workload: memory accounting's
  `benchmarks/remote_session/` SERVE=node with its engines on vllm's exact-int8 profile, batch-capped below the offered
  load, and every request's token times (`run.sh`, `engine_int8.py`, `summarize.py`); and the receipts pass part by
  part on an idle GPU (`pass_split.sh`).
