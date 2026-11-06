#!/usr/bin/env bash
# nci_cost.py on a CPU node: build nci_check and nci-phases (nci_phases/Main.lean, in a scratch Lake package against the
# checkout's nci_check), then run each CASE (suite, serve, glm; all three by default) into $OUT. Run from a clone (the
# build writes .lake), with GLM-5.3's rank 0 workload descriptor sent beside it for `glm` (README.md's fetch):
#   research run --project verity --on vy-nebius-cpu-2 --pool network --source . --cwd clone --mem-gb 320 --timeout 12h \
#     --send glm53-b1-rank0.descriptor.json.gz -- bash benchmarks/verifier_cost/nci_cost.sh [CASE...]
# NCI_COST_FLAGS (`--env`) goes to each nci_cost.py, e.g. "--no-policy --word-only".
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
L=$PWD/verity/core/protocols/network/nci/lean
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
(cd "$L" && lake build) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
mkdir -p "$S/phases"
cp benchmarks/verifier_cost/nci_phases/Main.lean "$L/lean-toolchain" "$S/phases/"
cat > "$S/phases/lakefile.toml" <<EOF
name = "nci_phases"
defaultTargets = ["nci-phases"]

[[require]]
name = "nci_check"
path = "$L"

[[lean_exe]]
name = "nci-phases"
root = "Main"
EOF
(cd "$S/phases" && lake build nci-phases) >> "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
cases=("$@")
[ ${#cases[@]} -gt 0 ] || cases=(suite serve glm)
rc=0
read -ra flags <<< "${NCI_COST_FLAGS:-}"
for c in "${cases[@]}"; do
  a=("${flags[@]}")
  [ "$c" = glm ] && a+=(--workload "$IN/glm53-b1-rank0.descriptor.json.gz")
  /usr/bin/time -v uv run python benchmarks/verifier_cost/nci_cost.py "$c" --out "$O" --phases "$S/phases/.lake/build/bin/nci-phases" \
    --scratch "$S" "${a[@]}" 2> >(tee "$O/$c.stderr" >&2) || rc=1
done
exit $rc
