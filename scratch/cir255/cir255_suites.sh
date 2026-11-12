#!/bin/bash
# CIR-255, #9390 at its head (the run's source): the suites that read what the change does, --fresh; repository first, alone.
set -u
uv sync -q --all-packages --extra torch-cpu || exit 1
echo "head $(git rev-parse HEAD 2>/dev/null)"
date -u +"repository start %FT%TZ"
uv run tools/verity/check/suites.py repository --fresh --mem-gb 120 --json "$OUT/suites-repository.json" --logs "$OUT/repository-logs"
echo "repository rc=$?"
date -u +"suites start %FT%TZ"
uv run tools/verity/check/suites.py verity-config verity-cli tools/verity/circuit_check verity/definitions --fresh --mem-gb 120 \
  --json "$OUT/suites.json" --logs "$OUT/suites-logs"; echo "suites rc=$?"
date -u +"vllm start %FT%TZ"
uv run tools/verity/check/suites.py adapters/vllm --fresh --mem-gb 120 --json "$OUT/suites-vllm.json" --logs "$OUT/suites-vllm-logs" \
  -- -k settings; echo "vllm rc=$?"
date -u +"end %FT%TZ"
