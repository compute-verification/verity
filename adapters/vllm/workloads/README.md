# The vLLM integration's row workloads

A row's workload file is a fixture in the evidence store, not in the tree. The tree keeps each one's name and digest
here, in `SHA256SUMS`, and its id in `fixtures/artifacts.json`.

## Interface

- **Provides:** `SHA256SUMS`, the digest each fetched workload file is held to.
- **Loaded by:** the integration, through `verity_vllm.config`, which names the directory (`WORKLOADS`) and fetches and
  checks a file (`workload_file`). By path, `demos/vllm`, the scripts of `benchmarks/prove_path`,
  `benchmarks/one_stage` and `benchmarks/pous` that pass a workload file, and `check`'s neutrality tier and public
  boundary (`tools/verity/check`).
- **Loads:** nothing.
