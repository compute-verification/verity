#!/usr/bin/env bash
# nci_glm on a CPU node: build `nci-check`, fetch a vLLM Build's workload Programs (every rank's) from the store and run
# nci_glm.py into $OUT. Run from a clone (the build writes .lake), with a store key (--store-read):
#   research run --on auto --store-read --cwd clone --cpus 16 --mem-gb 256 --max-min 240 -- \
#     bash benchmarks/verifier_cost/nci_glm.sh art:32c488d8ca7bb12f80b38e45ec5a97d95312bcb62974a4d776da0b58d1adec35 --jobs 8
# Arguments after ART go to nci_glm.py.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
art=${1:?usage: nci_glm.sh ART [ARGS...]}
shift
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
(cd verity/core/protocols/network/nci/lean && lake build nci-check) > "$O/build.log" 2>&1 || { tail -40 "$O/build.log" >&2; exit 1; }
uv sync -q --package research
uv run --no-sync research data fetch "$art" --to "$d" \
  --path 'build/rank*/build_workload/descriptor.json.gz' --path 'build/rank*/build_workload/workload_program.json' > "$O/fetch.log"
NCI_GLM_BUILD=$art uv run --no-sync python benchmarks/verifier_cost/nci_glm.py --build "$d" --out "$O" "$@"
