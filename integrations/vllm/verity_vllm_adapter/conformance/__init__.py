"""Configuration-conformance harness (lane CH, `docs/vllm-poc/coverage-wave-2026-09-07.md` Part 2).

A vLLM configuration is a *profile port*: a directory `vllm-poc/profiles/<port>/` with exactly five inputs
(`vllm_config.json`, `static_config.py`, `bindings.json`, `capture_mapping.py`, `expectations.json`). This
package turns a port into the standard conformance outputs under `fixtures/results/CONF-<port>-<utc>/`
(`production_repro.json`, `replay_repro.json`, `program.json`, `correspondence.json`, `missing_values.json`,
`mismatches.jsonl`, `descriptor_stats.json`, `level.json`, `ladder.json`, plus a human `CONFORMANCE.md`) and a
verdict on the promotion ladder **C0 runs · C1 replay · C2 representation · C3 protocol · C4 proof · C5 vertical
slice** (`ladder.py`; C1–C4 are `level.py`'s Level 1–4 under their public names, both spellings accepted on input)
that is never claimed above the evidence present. Unsupported outcomes are first-class and closed-enum
(`taxonomy.py`).

The harness only *calls* existing modules: `verity_vllm_adapter.{production,replay,compare,verify}` via
`pod/run_case.sh` (GPU stages), `verity_vllm.correspondence.core` (Level 2), `verity_vllm_adapter.profile` (the
profile manifest), the typed-obligation export and the SP1 host records (Level 4 pointers). A port must not
add code paths here; where a seam is missing it is added behind an option a port declares in
`vllm_config.json` (`parallelism.tensor_parallel_size`, `adapter.capture_per_rank`).

Vocabulary (GUIDANCE §13.3): `Program`, `FunctionDefinition`, `FunctionCall`, `Gate`, `Value`, `Input` gate,
`ExecutionTranscript`, static configuration, registered primitive family. Backend nouns stay behind the
lowering boundary.

    python -m verity_vllm_adapter.conformance validate --port b1-eager
    python -m verity_vllm_adapter.conformance run --port b1-eager --out DIR --from-results RESULTS_DIR ...
    python -m verity_vllm_adapter.conformance reproduce --port b1-eager        # committed artefacts, no GPU
    python -m verity_vllm_adapter.conformance matrix                            # docs/vllm-poc/conformance-matrix.md
"""

from __future__ import annotations

SCHEMA_PORT = "verity-vllm/profile-port/v0"
SCHEMA_EXPECTATIONS = "verity-vllm/port-expectations/v0"
SCHEMA_BINDINGS = "verity-vllm/port-bindings/v0"
SCHEMA_LEVEL = "verity-vllm/conformance-level/v0"
SCHEMA_LADDER = "verity-vllm/conformance-ladder/v0"
SCHEMA_RUN = "verity-vllm/conformance-run/v0"

OUTPUT_FILES = (
    "production_repro.json",
    "replay_repro.json",
    "program.json",
    "correspondence.json",
    "missing_values.json",
    "mismatches.jsonl",
    "descriptor_stats.json",
    "level.json",
    "ladder.json",
)
