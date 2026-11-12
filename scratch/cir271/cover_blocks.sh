set -uo pipefail
mkdir -p "$OUT"
echo "BLOCKS tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') on $(hostname) from $(date -u +%H:%M:%SZ)"
export PYTHONPATH="$PWD:$PWD/benchmarks/numerical/python:$PWD/verity/protocols/flock/prover/python:$PWD/adapters/vllm:$PWD/tools/verity/circuit_check/src${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/time -v .venv/bin/python scratch/cir271/cover_blocks.py scratch/cir271/cover_blocks_extra.json "$OUT" > "$OUT/cover_blocks.log" 2> "$OUT/cover_blocks.err"
rc=$?
grep '^BLOCKS' "$OUT/cover_blocks.log" | cut -c1-300
tail -n 25 "$OUT/cover_blocks.err" | cut -c1-300
echo "BLOCKS rc=$rc at $(date -u +%H:%M:%SZ)"
exit $rc
