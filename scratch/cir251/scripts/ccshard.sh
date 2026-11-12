set -o pipefail
K="$1"
MEM="$2"
git rev-parse HEAD | tee "$OUT/head.txt"
uv sync -q --all-packages --extra torch-cpu || exit 3
mkdir -p "$OUT/shard"
echo "jobs $(nproc)" | tee "$OUT/jobs.txt"
uv run --no-sync python -m circuit_check --all -q --jobs "$(nproc)" --out "$OUT/circuit-check.json" \
  --cache "$HOME/.cache/verity-check/circuit-check" --catalog-units \
  --mem-gb "$MEM" --shard "$K/4" --shard-dir "$OUT/shard" > "$OUT/circuit-check.txt" 2>&1
rc=$?
echo "shard $K/4 rc $rc" | tee "$OUT/rc.txt"
exit $rc
