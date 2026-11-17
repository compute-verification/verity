# The vLLM integration's workloads

A workload file is a fixture in the evidence store, not in the tree. The tree keeps each one's name and digest
here, in `SHA256SUMS`, and its id in `artifacts.json`. A workload `generation.toml` lists is also made from the tree alone:
`verity-vllm workload ROW`, given its checkpoint's tokenizer.json (the Hugging Face cache's snapshot or `--tokenizer`),
writes the file `SHA256SUMS` names, byte for byte.

## Interface

- **Provides:** `SHA256SUMS`, the digest each fetched workload file is held to, and `generation.toml`, what regenerating a
  row reads that its id doesn't give.
- **Loaded by:** the integration, through `verity_vllm.config`, which names the directory (`WORKLOADS`) and fetches and
  checks a file (`workload_file`). By path, the workload's scripts (`adapters/vllm/tools/workload`), the scripts of `benchmarks/prove_path`,
  `benchmarks/one_stage` and `benchmarks/pous` that pass a workload file, and `check`'s neutrality tier and public
  boundary (`tools/check`).
- **Loads:** nothing.
