#!/usr/bin/env bash
# verifier_cost on a CPU node: build the verifier COMMAND measures (`measure`: the NCI check, `nci-check`; `partition`:
# the one-stage pieces, `sampled-proofs-difftest`), run `run.py COMMAND` into $OUT, and put its results in the store as
# one evidence/v1 record, its art: id in $OUT/art.txt. Run from a clone (the build writes .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/job.sh measure --unrolled-max 40000000
# Arguments after COMMAND go to run.py COMMAND. Then: `verity bench verifier_cost plot ART --to PNG`.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
cmd=${1:?usage: job.sh measure|partition [ARGS...]}
shift
case $cmd in
  measure) pkg=verity/core/protocols/network/nci/lean exe=nci-check file=results.json schema=verity/verifier-cost/v1 ;;
  partition) pkg=verity/core/service/sampled_proofs exe=sampled-proofs-difftest file=partition.json
    schema=verity/verifier-cost-partition/v1 ;;
  *) echo "job.sh: COMMAND is measure or partition, not $cmd" >&2; exit 2 ;;
esac
uv sync -q --all-packages --extra torch-cpu
(cd "$pkg" && lake build "$exe") > "$O/build.log" 2>&1 || { tail -40 "$O/build.log" >&2; exit 1; }
rc=0
uv run python benchmarks/verifier_cost/run.py "$cmd" --out "$O" "$@" || rc=$?
uv run research data put --kind evidence/v1 --file "$O/$file" --preserve \
  --meta "{\"benchmark\": \"verifier_cost\", \"schema\": \"$schema\", \"verifier\": \"$exe\"}" > "$O/art.txt"
cat "$O/art.txt"
exit $rc
