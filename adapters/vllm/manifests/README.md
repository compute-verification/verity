# The vLLM integration's manifests

Run definitions the integration reads by path. No code lives here.

## Interface

- **Provides:** the checkpoint pins (`checkpoints.json`, `checkpoints-int8.json`), the index of the rows' records
  (`records.json`), the models each Definition module exists for (`definitions.json`), the int8 scales
  (`int8-k-scales.json`), the capture maps (`capture-maps/`) and the semantic profiles (`semantic-profiles/`).
- **Loaded by:** the integration, through `verity_vllm.config`, which names the directory (`MANIFESTS`), the checkpoint
  pins and the records' index. By path, the benchmarks and setup scripts that read the checkpoint pins or a capture map:
  `benchmarks/hash/dither`, `verity/protocols/spot/device/live_challenger`, `benchmarks/pous`, `benchmarks/replay_ladder`,
  `benchmarks/route_capture`, `benchmarks/one_stage` and `benchmarks/harness`. A test reads the semantic profiles.
- **Loads:** nothing.
