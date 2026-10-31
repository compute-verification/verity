#!/usr/bin/env bash
# The population-order check's cost against a template population's units: population.py writes, per N, a population of
# about N units over a layered call hierarchy (defaults: 234 root batches of 234 templates over six levels of at most 1,800
# composites, about 8,100 Definitions), and population-time (population/Main.lean, built in a scratch Lake package
# against the checkout's verifier) times `Partition.unitGates`, `Slots.popOk` and the per-unit reference
# (`TemplateQuery.gates` on every unit), refusing where the reference's gates differ. The reference runs only up to
# REF_MAX units (default 50). Run from a clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --cwd clone --mem-gb 90 -- \
#     bash benchmarks/verifier_cost/population.sh 20 50 100000 1000000 6963296
# Extra arguments to population.py go in POPULATION_ARGS. $OUT gets population.jsonl (one row per N and function:
# units, ms, peak RSS) and inputs.jsonl.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: population.sh N..." >&2; exit 2; }
L=$PWD/verity/core/service/sampled_proofs
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/time"
cp benchmarks/verifier_cost/population/Main.lean "$L/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "population_time"
defaultTargets = ["population-time"]

[[require]]
name = "core_sampled_proofs"
path = "$L"

[[lean_exe]]
name = "population-time"
root = "Main"
EOF
(cd "$S/time" && lake build population-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
# shellcheck disable=SC2086
uv run python benchmarks/verifier_cost/population.py "$S/in" ${POPULATION_ARGS:-} "$@" | tee "$O/inputs.jsonl"
for n in "$@"; do
  ref=(); [ "$n" -le "${REF_MAX:-50}" ] || ref=(--no-ref)
  "$S/time/.lake/build/bin/population-time" "$S/in/pop-$n.json" "${ref[@]}" | grep '^PHASE ' | sed 's/^PHASE //' \
    | sed "s/^{/{\"n\":$n,/" | tee -a "$O/population.jsonl"
done
