#!/usr/bin/env bash
# nci_glm.py on a CPU node: build nci-check, then check GLM-5.3's Programs into $OUT, the instances in a scratch
# directory. Run from a clone (the build writes .lake), with what the cases read sent beside it (README.md): the Build's
# eight workload descriptors and its meta, or the registered node's descriptor and its registration (`registered-node`):
#   research run --project verity --on vy-nebius-cpu-2 --source . --cwd clone --mem-gb 420 --mem-wait 3600 --timeout 6h \
#     --send glm53-b1-pins.json $(for r in 0 1 2 3 4 5 6 7; do echo --send glm53-b1-rank$r.descriptor.json.gz; done) \
#     -- bash benchmarks/verifier_cost/nci_glm.sh --jobs 10 [ARGS...]
#   research run --project verity --on vy-nebius-cpu-4 --source . --cwd clone --mem-gb 120 --timeout 16h \
#     $(for f in descriptor.json.gz trusted.json provenance.json registration.json; do echo --send glm53-b1-node.$f; done) \
#     -- bash benchmarks/verifier_cost/nci_glm.sh --case registered-node --node-digest HEX --jobs 2 [ARGS...]
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
a=()
if [ -e "$IN/glm53-b1-pins.json" ]; then
  a+=(--pins "$IN/glm53-b1-pins.json" --rank "$IN"/glm53-b1-rank{0..7}.descriptor.json.gz)
fi
if [ -e "$IN/glm53-b1-node.descriptor.json.gz" ]; then
  a+=(--node "$IN/glm53-b1-node.descriptor.json.gz")
  for f in trusted provenance registration; do
    if [ -e "$IN/glm53-b1-node.$f.json" ]; then a+=(--node-$f "$IN/glm53-b1-node.$f.json"); fi
  done
fi
uv sync -q --all-packages --extra torch-cpu
(cd verity/core/protocols/network/nci/lean && lake build nci-check) > "$O/build.log" 2>&1 || { tail -40 "$O/build.log" >&2; exit 1; }
uv run python benchmarks/verifier_cost/nci_glm.py measure --out "$O" --scratch "$d" "${a[@]}" "$@"
