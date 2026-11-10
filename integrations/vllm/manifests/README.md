# The vLLM integration's manifests

Run definitions the integration reads by path. No code lives here.

## Interface

- **Provides:** the checkpoint pins (`checkpoints.json`, `checkpoints-int8.json`), the index of the rows' records
  (`records.json`), the models each Definition module exists for (`definitions.json`), the int8 scales
  (`int8-k-scales.json`), the capture maps (`capture-maps/`) and the semantic profiles (`semantic-profiles/`).
- **Loaded by:** `verity_vllm.config`, which names the directory (`MANIFESTS`), and the modules that take a file from it:
  `glue`, `commit.capture_map`, `acquire.sources.route_source`, `engine.int8_k_scales` and
  `program.predict.step_needs`. A test reads the semantic profiles.
- **Loads:** nothing.
