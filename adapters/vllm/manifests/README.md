# The vLLM integration's manifests

Run definitions the integration reads by path. No code lives here.

## Interface

- **Provides:** the checkpoint pins (`checkpoints.json`, `checkpoints-int8.json`), the index of the workloads' records
  (`records.json`), the models each Definition module exists for (`definitions.json`), the int8 scales
  (`int8-k-scales.json`) and the capture maps (`capture-maps/`).
- **Loaded by:** the integration, through `verity_vllm.config`, which names the directory (`MANIFESTS`), the checkpoint
  pins and the records' index. By path, the benchmarks and setup scripts that read the checkpoint pins or a capture map:
  `benchmarks/hash/dither`, `verity/protocols/compose/verifier/live_challenger`, `benchmarks/pous`, `benchmarks/replay_ladder`,
  `benchmarks/route_capture`, `benchmarks/one_stage` and `benchmarks/harness`.
- **Loads:** nothing.
