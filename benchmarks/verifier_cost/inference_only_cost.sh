#!/usr/bin/env bash
# inference_only_cost.py on a CPU node: build inference_only_check and inference-only-phases (inference_only_phases/Main.lean, in a scratch Lake package against the
# checkout's inference_only_check), then run each CASE (suite, serve, glm; all three by default) into $OUT. Run from a clone (the
# build writes .lake), with GLM-5.3's rank 0 workload descriptor sent beside it for `glm` (README.md's fetch):
#   research run --project verity --on vy-nebius-cpu-2 --pool network --source . --cwd clone --mem-gb 320 --timeout 12h \
#     --send glm53-b1-rank0.descriptor.json.gz -- bash benchmarks/verifier_cost/inference_only_cost.sh [CASE...]
# INFERENCE_ONLY_COST_FLAGS (`--env`) goes to each inference_only_cost.py, e.g. "--no-policy --word-only".
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
L=$PWD/verity/protocols/policy/inference_only/verifier/lean
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
(cd "$L" && lake build) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
mkdir -p "$S/phases"
cp benchmarks/verifier_cost/inference_only_phases/Main.lean "$L/lean-toolchain" "$S/phases/"
cat > "$S/phases/lakefile.toml" <<EOF
name = "inference_only_phases"
defaultTargets = ["inference-only-phases"]

[[require]]
name = "inference_only_check"
path = "$L"

[[lean_exe]]
name = "inference-only-phases"
root = "Main"
EOF
(cd "$S/phases" && lake build inference-only-phases) >> "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
cases=("$@")
[ ${#cases[@]} -gt 0 ] || cases=(suite serve glm)
rc=0
read -ra flags <<< "${INFERENCE_ONLY_COST_FLAGS:-}"
for c in "${cases[@]}"; do
  a=("${flags[@]}")
  [ "$c" = glm ] && a+=(--workload "$IN/glm53-b1-rank0.descriptor.json.gz")
  /usr/bin/time -v uv run python benchmarks/verifier_cost/inference_only_cost.py "$c" --out "$O" --phases "$S/phases/.lake/build/bin/inference-only-phases" \
    --scratch "$S" "${a[@]}" 2> >(tee "$O/$c.stderr" >&2) || rc=1
done
exit $rc
