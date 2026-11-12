#!/usr/bin/env bash
# The proof-block census of a deferred TP Commit's Build: every rank's Programs of record over every request step
# (`m1_families.py --steps all`), each rank's families sized (`--estimate`), then `block_census.py` over all ranks.
# Run on the machine holding the row directory (its build/rank<r>/ and commit/), as a research run of a clone:
#   research run --on MACHINE --project verity --source . --cwd clone --store-read --mem-gb 96 -- \
#     bash benchmarks/one_stage/block_census.sh ROW_DIR
#   BC_JOBS=4 (ranks at once, about 11 GB each)
set -uo pipefail
ROW=${1:?the row directory: build/rank<r>/ and commit/}
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"; JOBS=${BC_JOBS:-4}
WL=adapters/vllm/workloads/$(basename "$ROW").json
export PATH=$HOME/.local/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD; echo "$ROW"; } | tee "$O/machine.txt"
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
RANKS=$(cd "$ROW/build" && ls -d rank* | sed 's/rank//' | sort -n)
(cd "$ROW" && find commit -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum) > "$O/commit.sha256"
(cd "$ROW" && find build -type f -path 'build/rank*' -print0 | sort -z | xargs -0 sha256sum) > "$O/build.sha256"
cp "$ROW/commit/versions.json" "$O/" 2>/dev/null
$PY -c "from verity_vllm.pipeline.workload_file import workload_file; print(workload_file('$WL'))" || exit 4
rank() {
  $PY benchmarks/one_stage/m1_families.py --commit "$ROW/commit" --build-root "$ROW/build" --workload "$WL" --rank "$1" \
    --steps all --out "$O/census-rank$1.json" > "$O/census-rank$1.log" 2>&1 || return 6
  $PY benchmarks/one_stage/m1_families.py --estimate "$O/census-rank$1.json" --out "$O/sized-rank$1.json" \
    > "$O/sized-rank$1.log" 2>&1 || return 7
}
pids=()
for r in $RANKS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  rank "$r" & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ $rc -eq 0 ] || { for f in "$O"/*-rank*.log; do echo "== $f"; tail -5 "$f"; done; exit $rc; }
$PY benchmarks/one_stage/block_census.py --sized $(for r in $RANKS; do echo "$O/sized-rank$r.json"; done) --out "$O" \
  2>&1 | tee "$O/block_census.log"
exit "${PIPESTATUS[0]}"
