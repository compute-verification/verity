#!/usr/bin/env bash
# verifier_cost on a CPU node: build the NCI check (`nci-check`), measure every config into $OUT (run.py measure), and put
# results.json in the store as one evidence/v1 record, its art: id in $OUT/art.txt. Run from a clone (the build writes .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/job.sh --unrolled-max 40000000
# Arguments go to run.py measure. Then: `verity bench verifier_cost plot ART --to PNG`.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
uv sync -q --all-packages --extra torch-cpu
(cd verity/core/protocols/network/nci/lean && lake build nci-check) > "$O/build.log" 2>&1 || { tail -40 "$O/build.log" >&2; exit 1; }
rc=0
uv run python benchmarks/verifier_cost/run.py measure --out "$O" "$@" || rc=$?
uv run research data put --kind evidence/v1 --file "$O/results.json" --preserve \
  --meta '{"benchmark": "verifier_cost", "schema": "verity/verifier-cost/v1", "verifier": "nci-check"}' > "$O/art.txt"
cat "$O/art.txt"
exit $rc
